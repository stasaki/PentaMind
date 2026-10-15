![PentaMind Logo](https://github.com/stasaki/PentaMind/raw/main/pentamind-logo.png)

--------------------------------------------------------------------------------

PentaMind is a deep learning model that predicts **global cognition** from a scanned drawing of
**intersecting pentagons** (the pentagon copying item of the MMSE). It was trained on 13,777
drawings from 3,111 older adults in the ROS, MAP and MARS cohorts of the Rush Alzheimer's
Disease Center (RADC). It is implemented in PyTorch.

- Paper: [Explainable deep learning approach for extracting cognitive features from hand-drawn images of intersecting pentagons](https://www.nature.com/articles/s41746-023-00904-w),
  Tasaki S, Kim N, Truty T, Zhang A, Buchman AS, Lamar M, Bennett DA. *npj Digital Medicine* 6, 157 (2023). doi:10.1038/s41746-023-00904-w
- Model weights: [huggingface.co/stasaking/pentamind](https://huggingface.co/stasaking/pentamind)

## Contents
- [What the model does](#what-the-model-does)
- [Quick start](#quick-start)
- [Input images](#input-images)
- [Using PentaMind on new data](#using-pentamind-on-new-data)
- [Limitations and cautions](#limitations-and-cautions)
- [Reproducing the paper (original scripts)](#reproducing-the-paper-original-scripts)
- [Citation](#citation) · [License](#license)

## What the model does

| | |
|---|---|
| Input | One image of a pentagon drawing, resized to 224 x 224 pixels, RGB |
| Output | One number: predicted **global cognition** (higher = better cognition) |
| Architecture | VGG19 with batch normalization (torchvision `vgg19_bn`), ImageNet start, last layer replaced by 1 output (regression) |
| Released models | 10 models (`vgg19_bn-1` ... `vgg19_bn-10`). The final score is the **average of the 10 models** |
| Performance (paper) | Out-of-fold, PentaMind explained 23.3% of the variance in global cognition, 1.9 times more than the usual pass/fail rating of the drawing |

**Global cognition** in ROS/MAP/MARS is a composite of 19 cognitive tests (the MMSE and the
pentagon item are not part of it), measured at the same research visit as the drawing. It is on a z-score scale: 0 is about the cohort average, and
one unit is about one standard deviation of the individual tests.

**The 10 models.** The 10 models come from 10-fold cross-validation. The participants were split
into 10 groups; each model was trained on 9 groups and left one group out. Averaging the 10 models
gives a more stable score than any single model.

## Quick start

Requires Python 3.9 or newer. A GPU is optional. On a small 2-core CPU, scoring takes about
5 seconds per image for all 10 models; on a GPU it is much faster.

```bash
git clone https://github.com/stasaki/PentaMind.git
cd PentaMind
pip install -r requirements.txt

# score every image in a folder; the 10 models (5.6 GB) are downloaded from Hugging Face on first use
python pentamind_predict.py --images data/images/ --out scores.csv
```

`scores.csv` has one row per image:

| Column | Meaning |
|---|---|
| `file_name` | image file name |
| `pentamind_score` | mean of the 10 models = predicted global cognition (higher = better) |
| `sd_models` | standard deviation across the 10 models (high = models disagree) |
| `model_1` ... `model_10` | each model's score (only with `--per_model`) |

Useful options:

| Option | What it does |
|---|---|
| `--crop` | crop each image to the drawing (dark pixels) first. Use this for whole-page scans |
| `--n_tta 30` | test-time augmentation as in the paper (see below). 30 times slower |
| `--features feats.npz` | also save image features (`gap4`, `gap5`: 512 numbers; `fc2`: 4096 numbers) from each of the 10 models, to train your own model |
| `--models_dir models/` | use weight files you already downloaded |

**Test-time augmentation (TTA).** In the paper, each image was scored 30 times: once as it is and
29 times with random changes (flips, 90-degree rotations, small shifts and rotations, contrast,
blur, sharpening). The 30 scores were averaged. This makes the score less sensitive to small
differences in image quality. `--n_tta 1` (default) scores only the original image; it is faster
and usually very close.

## Input images

The training images were made like this:
1. The pentagon drawing was scanned from the paper test form.
2. The region with the pentagons was cut out (clipped), so the drawing fills most of the image.
3. The image was padded with white to a 500 x 500 square, then resized to 224 x 224.

So the model expects: **dark lines on a white background, the drawing centered and filling most
of a square image, nothing else on the page** (no text, boxes or signatures).

`pentamind_predict.py` converts any image to RGB on white, pads it to a square and resizes it to
224 x 224. With `--crop`, it first cuts out the box around the dark pixels (plus a 10% margin).
`data/images/` contains synthetic example images in the right format.

## Using PentaMind on new data

**New pentagon drawings (other cohorts).** Use `pentamind_predict.py` as above. Check a few
images by eye after cropping. The score is on the ROS/MAP/MARS global cognition scale; in a new
cohort, the absolute values may be shifted, so it is safer to use the score for ranking or as a
variable in a regression than as an absolute cognition value.

**Other drawings (clocks, cubes, spirals, ...).** PentaMind has only seen pentagons. Its score on
other figures is "out of domain" and is usually only weakly related to cognition. A better
approach is usually: extract features with `--features`, then fit a simple model (for example
ridge regression with cross-validation) on your own labels.

**Digital pen data.** PentaMind uses only the final image, not time, pressure or stroke order. To
use it with digital pen recordings:
1. Draw the strokes as black lines on a white background (for example with Pillow or matplotlib).
2. Use a line width similar to a pen on paper. In the paper's simulation, very thin lines lowered
   the score, so check that your results do not depend strongly on the line width you choose.
3. Crop to the drawing and pad to a square (`--crop` does this).
4. Use the PentaMind score or features **together with** timing and pressure features; they
   measure different things.

**What changes the score (from the paper's simulation).** The score went down with: the wrong
number of corners, no overlap or too much overlap between the two pentagons, irregular or unequal
shapes, small drawings, very thin lines, and wavy (shaky) lines. Wavy lines lowered the score even
when the shapes were correct.

## Limitations and cautions

- **Do not use the 10-model average on ROS, MAP or MARS drawings to measure performance.** Every
  RADC drawing in the training data was used to train 9 of the 10 models, so results on those
  drawings will look better than they really are. If your data includes RADC participants,
  contact us.
- **Not a diagnostic tool.** PentaMind is for research. It has not been validated for clinical use.
- **Training population.** Older adults in the US (mostly White; ROS and MAP enrolled people
  without dementia at baseline; MARS enrolled African American participants), pencil-and-paper
  drawings, one scanning setup. Performance on other populations, languages, forms and scanners
  is unknown.
- **Modest accuracy.** The score explains about a quarter of the variance in global cognition.
  Predictions are pulled toward the average (the range of scores is narrower than the range of
  true cognition).
- **Image format matters.** Page borders, printed text, very large margins or a drawing that is
  too small in the frame can change the score. Look at your images after preprocessing.

## Reproducing the paper (original scripts)

The original scripts used in the paper are kept in `functions/`. They need an NVIDIA GPU and the
conda environment from 2021 (Python 3.7, PyTorch 1.10, CUDA 11.1):

```bash
conda env create -f environments/environment.yaml -n pentamind
conda activate pentamind
```

- Train a PentaMind model:
```bash
python functions/PentaMind.py --model_name vgg19_bn --use_pretrained True --pheno_file_dir data/phenotype/ --image_file_dir data/images/ --output_dir results/PentaMind/ --optim sgd --batch-size 32 --epochs 90 --lr 0.001 --momentum 0.9 --weight-decay 0.0001
```

- Predict with one model:
```bash
python functions/PentaMind-pred_ave.py --model_name vgg19_bn --model_loc models/vgg19_bn-1_model.pth --pheno_file data/phenotype/test.txt --image_file_dir data/images/ --output_dir results/PentaMind-pred_ave/ --batch-size 32
```
  The output `y_pred.npy` has shape (number of images, 30): column 1 is the original image and
  columns 2-30 are the 29 augmented copies (see TTA above). Average over the 30 columns to get the
  score of this model, then average over the 10 models. `pentamind_predict.py` does all of this
  for you.

- Intersecting pentagons simulator: open `functions/simulate_pentagon.ipynb` in
  [Jupyter](https://jupyter.org/), then plot the results with `functions/simulate_pentagon.Rmd`
  in [RStudio](https://posit.co/download/rstudio-desktop/). Note: the simulator scores each image
  once, after one random augmentation (seed 1234).

## Dataset

All data in the `data/` directory is **synthetic**. It only shows the file format (`data/phenotype/*.txt`:
tab-separated, columns `file_name` and `value`). The real pentagon images and cognitive scores used
in the study can be requested from the [RADC Resource Sharing Hub](https://www.radc.rush.edu/)
under a data use agreement.

## Citation

If you use PentaMind, please cite:

```bibtex
@article{tasaki2023pentamind,
  title   = {Explainable deep learning approach for extracting cognitive features from hand-drawn images of intersecting pentagons},
  author  = {Tasaki, Shinya and Kim, Namhee and Truty, Tim and Zhang, Ada and Buchman, Aron S. and Lamar, Melissa and Bennett, David A.},
  journal = {npj Digital Medicine},
  volume  = {6},
  pages   = {157},
  year    = {2023},
  doi     = {10.1038/s41746-023-00904-w}
}
```

## License

- **Code**: [BSD 3-Clause License](LICENSE).
- **Model weights** (trained on ROS/MAP/MARS data): [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
  (attribution, **non-commercial**, share-alike).
- It is your responsibility to check that your use of the models is allowed.

## Acknowledgments

We thank all the participants of the ROS, MAP and MARS studies. Their contributions made this work
possible. We also thank the machine learning community for making pre-trained models easy to access.

## Contact

Questions and bug reports: please open a [GitHub issue](https://github.com/stasaki/PentaMind/issues).
