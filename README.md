# Voice Anti-spoofing: LCNN on ASVspoof 2019 LA

<p align="center">
  <a href="#about">About</a> •
  <a href="#results">Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#data">Data</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#method">Method</a> •
  <a href="#repository-structure">Repository Structure</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="/LICENSE">
   <img src="https://img.shields.io/badge/license-MIT-blue.svg">
</a>
</p>

## About

This repository contains a **countermeasure (CM) system for voice anti-spoofing**: given a
recording, the model decides whether it is genuine human speech (*bonafide*) or speech produced by
a text-to-speech or voice conversion system (*spoof*). It is trained and evaluated on the Logical
Access (LA) part of the [ASVspoof 2019 dataset](https://datashare.ed.ac.uk/handle/10283/3336)
([Kaggle Link](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset)), where the training
and development parts are built from one set of spoofing algorithms and the evaluation part from a
different and larger one. The task is therefore to generalise to attacks that were not seen during
training.

The model is a **Light CNN (LCNN)** with Max-Feature-Map activations. Its architecture follows
[Wu et al. (2015)](https://arxiv.org/abs/1511.02683), with the exact layer configuration taken from
the Speech Technology Center system of
[Lavrentyeva et al. (2019)](https://arxiv.org/abs/1904.05576). The data preparation and training
recipe come from [Wang & Yamagishi (2021)](https://arxiv.org/abs/2103.11326).

The project is built on top of the
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template). The model was
trained with this code, and all logs come directly from it.

## Results

The model is evaluated on the evaluation part of ASVspoof 2019 LA with the `compute_eer` function
supplied with the assignment.

| System                                   | Front-end | Training criterion                                             | EER, %      |
| ---------------------------------------- | --------- | -------------------------------------------------------------- | ----------- |
| **This work** (`checkpoint-epoch44`)     | LFCC      | cross-entropy, class-weighted                                  | **6.07**    |
| Lavrentyeva et al. (2019), `LFCC-LCNN`   | LFCC      | A-Softmax                                                      | 5.06        |
| Wang & Yamagishi (2021), `LCNN-trim-pad` | LFCC      | sigmoid (= cross-entropy), 6 seeds                             | 2.54 – 3.47 |

The last row is the closest published comparison: the same architecture with the same front-end,
and the recipe here follows it. All four criteria in that paper are trained with cross-entropy and
differ only in the output activation, and for two classes the softmax reduces to a sigmoid, so
their `sigmoid` column is the configuration used here. The result reported here is behind it. Over
their whole grid the same architecture ranges from 2.31 % to 7.06 %, being sensitive to
initialisation; only one seed was trained here.

At the end of training the model reaches 0.59 % EER on the development partition against the
6.07 % above — the difference between seen and unseen attacks described earlier. Lavrentyeva et al.
report the same pattern, 0.157 % against 5.06 %.

Training took about 1 hour 45 minutes on a single Kaggle GPU — 50 epochs of 500 steps. Scoring
every saved checkpoint on the evaluation partition afterwards brought the whole session to roughly
2.9 hours.

Training and evaluation curves are published as a WandB report:
**[WandB Report](https://api.wandb.ai/links/gmbreed-hse-university/92ul2ckj)**

## Installation

The pinned dependencies are published for Python 3.9 through 3.12; the commands below use 3.10.
Create and activate an environment, either with `conda`:

```bash
conda create -n antispoof python=3.10 && conda activate antispoof
```

or with `venv`:

```bash
python3 -m venv antispoof_env && source antispoof_env/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

Two of them deserve a note. `torchaudio` carries both the FLAC reader and the LFCC front-end, so
nothing runs without it. `soundfile` is listed explicitly so that FLAC decoding works on a bare
installation, without depending on a system-wide FFmpeg or SoX.

The pre-commit hooks are optional and only matter if you intend to commit to the repository:

```bash
pre-commit install
```

## Data

The project uses the Logical Access partition of ASVspoof 2019, available either from
[Kaggle](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset) or from the
[Edinburgh DataShare mirror](https://datashare.ed.ac.uk/handle/10283/3336). There is no
preprocessing step: the code reads the original `.flac` files and the original protocol files as
they come.

The directory passed as `dataset_root` is expected to keep the layout of the archive:

```
LA
├── ASVspoof2019_LA_cm_protocols
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   └── ASVspoof2019.LA.cm.eval.trl.txt
├── ASVspoof2019_LA_train/flac
├── ASVspoof2019_LA_dev/flac
└── ASVspoof2019_LA_eval/flac
```

In the configs `dataset_root` points at a Kaggle path, since that is where the model was trained,
so on any other machine it has to be given on the command line — the examples below all do that.

## How To Use

All commands are run from the repository root.

### Training

```bash
python3 train.py -cn=asvspoof dataset_root=/path/to/LA
```

Checkpoints are written to `saved/lcnn_baseline/`, one every two epochs, together with
`model_best.pth` for the lowest EER on the development partition. The full config of the run is
saved next to them as `config.yaml`, so any result can be traced back to the settings that produced
it. Anything in the config can be overridden from the command line:

```bash
python3 train.py -cn=asvspoof dataset_root=/path/to/LA writer.run_name=my_run trainer.seed=7
```

A second config trains the same model with A-Softmax in place of weighted cross-entropy:

```bash
python3 train.py -cn=asvspoof_asoftmax dataset_root=/path/to/LA
```

### Inference

```bash
python3 inference.py -cn=inference_asvspoof dataset_root=/path/to/LA inferencer.from_pretrained=saved/lcnn_baseline/checkpoint-epoch44.pth
```

This prints the EER on the evaluation partition and writes the predictions to
`data/saved/asvspoof_eval/eval/submission.csv`: one `utterance_id,score` line per utterance, where
the score is the probability the model assigns to the *bonafide* class. That file is the submission
format the task asks for.

### Reproducing the reported result

The 6.07 % in the table above comes from `checkpoint-epoch44` of a run of `src/configs/asvspoof.yaml`
with its default seed of 1. Running the training command and then the inference command above
reproduces it, up to the nondeterminism of GPU training.

## Method

### Front-end

Every recording is turned into 20 linear-frequency cepstral coefficients, computed from a bank of
20 linear triangular filters over a 512-point FFT with a 20 ms window and a 10 ms hop, and joined
with their first and second differences, which gives 60 numbers per frame
(`src/transforms/lfcc.py`). This follows Wang & Yamagishi, except that they replace the first
coefficient with the log spectral energy.

### Fixed-length input

Every recording is cropped or zero-padded to 120 000 samples: 7.5 seconds at 16 kHz, or 750 frames
at a 10 ms hop, which the same paper reports covers 98 % of the database. The crop window is drawn
at random during training and fixed to the first window at evaluation, so a recording always
receives the same score. Short recordings are zero-padded, as in the reference recipe.

### Architecture

A Light CNN in the layer configuration of Table 1 of Lavrentyeva et al.: nine convolutions, each
followed by a Max-Feature-Map activation, with four max-pooling layers and batch normalisation in
between, then a fully connected layer with another Max-Feature-Map, dropout of 0.75, a final batch
normalisation and the classifier (`src/model/lcnn.py`). Max-Feature-Map takes the element-wise
maximum of two halves of the channel axis, halving the channel count at every activation. The model
has 865 058 parameters.

### Training

Cross-entropy with class weights of 1 and 9, offsetting the roughly one-to-nine ratio of bonafide
to spoofed recordings. A-Softmax is implemented as well (`src/loss/a_softmax.py`) and is selectable
through the `asvspoof_asoftmax` config.

AdamW with a learning rate of 3e-4 and no weight decay, which makes it plain Adam; the learning
rate is halved every 5 000 steps. Training runs for 50 epochs of 500 steps with a batch size of 32.
Features are not normalised, following the same recipe. The EER is computed over all the scores of
a partition at once rather than per batch (`src/metrics/eer_metric.py`).

## Repository Structure

The layout is the one the template comes with; the task-specific code lives in the directories it
leaves open.

```
src
├── configs      Hydra configs; asvspoof.yaml is the training config used here
├── datasets     protocol parsing, the trim-pad scheme, collation
├── transforms   the LFCC front-end, plus a log-spectrogram kept as a fallback
├── model        the Light CNN and the Max-Feature-Map activation
├── loss         weighted cross-entropy, and A-Softmax as an alternative
├── metrics      EER, and the accumulator that computes it over a partition
├── trainer      the training loop and the inferencer
├── logger       WandB and Comet ML writers
└── utils        initialisation and IO helpers
train.py         entry point for training
inference.py     entry point for inference and for writing the submission
```

## Credits

This repository is a variant of the
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template) by
[Blinorot](https://github.com/Blinorot), which is itself based on
[pytorch-template](https://github.com/victoresque/pytorch-template) and
[asr_project_template](https://github.com/WrathOfGrapes/asr_project_template).

The work builds on three papers:

- X. Wu, R. He, Z. Sun, T. Tan. *A Light CNN for Deep Face Representation with Noisy Labels*,
  [arXiv:1511.02683](https://arxiv.org/abs/1511.02683) — the architecture and the Max-Feature-Map
  activation.
- G. Lavrentyeva, S. Novoselov, T. Andzhukaev, M. Volkova, A. Gorlanov, A. Kozlov. *STC
  Antispoofing Systems for the ASVspoof2019 Challenge*,
  [arXiv:1904.05576](https://arxiv.org/abs/1904.05576) — the layer configuration used here.
- X. Wang, J. Yamagishi. *A Comparative Study on Recent Neural Spoofing Countermeasures for
  Synthetic Speech Detection*, [arXiv:2103.11326](https://arxiv.org/abs/2103.11326) — the data
  preparation and the training recipe.

## License

The template is distributed under the MIT license, and its copyright notice is kept unchanged in
[LICENSE](/LICENSE).

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
