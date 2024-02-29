# LLM_Compression

## Introduction

Welcome to our GitHub page! In this repository, we provide all files for training our models, performing inference, and obtaining evaluation results. We provide a brief overview of our project, but also elaborate on how each component was developed from a technical perspective. To learn more about our overall findings, please consult our project final report.

## Problem Statement and Hypothesis:

Our research concerns LLM inference speed. We notice how just a few huge transformer blocks can lead to highly accurate token prediction, but at the cost of adding a huge number of parameters and large depth, effecively increasing inference latency due to these computational costs. Given that an LLM is autoregressive, this latency can build up linearly.

Our key question we ask ourselves is: how can we reduce inference latency through architectural changes while maintaining evaluation results?

We hypothesize that through creating richer transformer blocks by adding nonlinear layers on each final attention head before concatention, we can reduce the total number of transformer blocks utilized, decreasing latency and maintaining accuracy.

## Implementation

### Procedure:
To create a controlled experiment, we propose models that have ~100M parameters. We train baselines from scratch instead of using preexisting ones to match the same model arguments for the trained novel models. Below are the steps we take to conduct our entire project:

1. We first pull and adapt code from Meta Research's Llama GitHub repository (LLama 2), which can be found [here](https://github.com/facebookresearch/llama).
   * The above code (inside the llama folder in their repository) only works for inference. We begin by modifying the model files to remove KV-caching (we cannot have a KV-cache during training) and downscaling parameters to make training feasible. After these changes, the model can be created from scratch and trained.
1. We add our proposed layers into the model. This involves initiating the weights, and modifying the forward pass to multiply the attention heads with our layers.

1. We develop a training environment and framework around the Llama architectures
   * We create a generic trainer script to train any passed-in model configuration
   * We add different runner files, which are entry points into beginning the training for particular model configurations
1. We conduct training for **8 models** - 4 are baseline, and 4 are novel
   * We train models consisting of 5-8 transformer layers (inclusive) for each category (baseline, novel).
   * We save the following for each trained model:
        * A model checkpoint, which has the raw weights, optimizer information, and similar supporting details.
        * Losses. We downsample by 4, or record a loss every 4 batches (size 32).
        * Elapsed training time. This number is subject to some small bias because we only have 1 GPU, which we may have utilized at time to do evaluation while training occurred in parallel, potentially causing minor slowdowns.
1. Perform evaluation of all our trained LLMs.
   * We adapt the evaluation framework from [TRACE](https://github.com/BeyonderXX/TRACE) for obtaining the following metrics: BLEU, Rouge, + more, across different datasets
   * We create our own scripts to evaluate **perplexity** using another portion of the training dataset (not seen by the model)
   * We create additional scripts and plots for obtaining our metric of interest, **inference time**, alongside number of parameters, training time, and more.

### Directory Walkthrough
We provide a high-level walkthrough of how our code is structured. For more details, you can check the directories for even more information about structure (READMEs inside select relevant directories)

#### Significant Directories

* [scripts](./draft-files-v1/remote-assets/scripts/) - This contains our entire pipeline for training all our models (including adapted model architectures and classes), checkpoints, performing evaluation, and relevant logs and outputs. More details can be found in the folder's README.
* [visualizations](./draft-files-v1/visualizations/) - We provide a series of jupyter notebooks for parsing our output files, from logs of losses to times to csv files, and visualizing all evaluation results.


#### Other Directories
* [colab-canary](./draft-files-v1/colab-canary/) - This is a scatch directory we utilized for testing some initial models inside of Google Colab, before we had access to our A100. It is no longer needed, but we keep it in case we need to move back to colab at some point.
* [model-scaffold](./draft-files-v1/model-scaffold/) - This contains the original relevant Llama files we pulled from the Meta's repository. This is also not of importance, but we maintain it as an easy reference to the original code
* [novel](./draft-files-v1/novel/) - This is a small directory for quick testing of loading our evaluation for the model. Not of critical importance.

## How to Run:
First, please install the [requirements](./draft-files-v1/remote-assets/requirements.txt). We recommend using a python "venv". The python version we utilize in our VM is 3.8.16, though our code should work for newer versions (untested).

Before running any examples, please ensure that you have activated the venv (```source <VENV_NAME+PATH>/bin/activate```).

There are multiple components that can be run. Below we list them.
* Training of the models: This is as simple as executing the runner files we have provided. Simply change directories into the [scripts folder](./draft-files-v1/remote-assets/scripts/), and execute a runner.
   * An example: One of the 8+ runners is the [5 layer novel model](./draft-files-v1/remote-assets/scripts/runner_prototype_novel_5_layer.py). Adjacent to this file / in the scripts directory, execute ```python runner_prototype_novel_5_layer.py```.
      * This will (and should) fail. We have checks to prevent overwriting of existing logs and checkpoints, and we have existing logs pushed into this repository. To make this work, change the directories utilized (SAVE PATH, LOSSES PATH, TIMES PATH). You can also change the number of rows and more to train on, to essentially run a very small training job. Please note that folders to the provided paths must fully exist (i.e. our script does not create the full path desired if it does not exist, to prevent user error). We highly recommend using a GPU here, as otherwise this function can easily fail (we move the model to cuda and set up a distributed environment).
* All visualizations:
   * These are found in our visualizations directory already linked previously. Simply utilize the python file from the venv (in its bin directory) as the jupyter kernel, and execute the jupyter notebooks if desired using this kernel and inside the visualization directory. **The visualizations are already left as outputs inside all notebooks**. Our visualizations can be reproduced easily from a rerun, however.



The following require checkpoints/model weights (and such also requires a high end GPU to place the model on) - and great news, we have these checkpoints available in our repository, kudos to Git LFS (Large File Storage). You may need to install [lfs](https://git-lfs.com/). If you have trouble obtaining them on git, they can also be obtained by using rsync / scp across the [checkpoints](./draft-files-v1/remote-assets/scripts/checkpoints/) directory in our VM (with the provided lab Azure private credentials). Please contact us if any help is needed here, we are happy to help. Worst case, they can be generated exactly the same as they are (though training each can take up to a day for just one model), as we seed our training process for reproducibility.

NOTE: If you want to skip the checkpoint download (it is 14 GB+), you should be able to use a command like this (untested):
```export GIT_LFS_SKIP_SMUDGE=1``` and then
```git clone ...```
. If you need to later fetch the lfs checkpoints, use ```git lfs pull```

ANOTHER NOTE: If these lfs checkpoints are finnicky, you can pull from a previous commit (sitting on main branch), i.e. "ec34c40".

* Perplexity Evaluation Results (RAW):
   * Our perplexity runner can be found [here](./draft-files-v1/remote-assets/scripts/perplexity-runners/slimpj_ppx_runner.py).
   * It can be configured to run perplexities across all 8 models. For one model it will take roughly 3-4 minutes on 1 A100 to obtain the raw perplexities (before averaging performing mean).
   * Logs are already produced and provided in the 
* Evaluation Results:
   * All Jupyter notebooks inside the scripts folder utilize the checkpoints and load them for usage of sequence generation. To see sequences generated by one of our models, please peruse the [testQA.json](./draft-files-v1/remote-assets/scripts/testQA.json) file. This, among other keys, has "prompts," "results," and "labels", each pointing to a list of values. Each indexed entry corresponds to other lists' entries with the same index.
* To play around:
   * You can utilize the [test_generation.ipynb](./draft-files-v1/remote-assets/scripts/test_generation.ipynb) file and run the first 7 cells (no more after the cleanup cell) to see how the model reacts to certain prompts. You can vary the prompts provided as well. Again to do this, you will need a good GPU available and permissions to create a distributed environment.

Again, please feel free to reach out to us if you have any additional questions; we'd love to help!
