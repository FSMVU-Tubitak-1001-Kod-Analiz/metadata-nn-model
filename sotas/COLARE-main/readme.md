# COLARE: Commit Classification via Fine-grained Context-aware Representation of Code Changes

COLARE is specifically designed for commit maintenance activity classification and has demonstrated outstanding performance, surpassing the state-of-the-art by 7.24% in accuracy and 7.35% in macro F1 score. COLARE's approach involves presenting and comparing code changes at the hunk level, incorporating fine-grained features based on the categories of modified files, and aggregating this with the representation of commit messages.

## Environment

To set up the environment for this project, we recommend using Python 3.9 or higher versions. The main dependent environments are as follows:

- torch==1.13.0
- numpy==1.21.6
- scipy==1.7.3
- transformers==4.27.4
- ...

To build an experiment environment, use the following commands:

``` shell
conda create --name exp python=3.9 -y
conda activate exp
pip3 install -r requirements.txt
```

## Dataset

The `dataset` folder contains all the datasets used in our paper, which can be utilized for training or evaluating the model directly.

Please note that there are two versions of the `multi programming language` dataset, which are essentially the same. `multi-lang-exp` is used in the experiments, while `multi-lang-annotated` contains the same data as multi-lang-exp, but with clearer field names. If you intend to use our provided dataset for other purposes, we recommend using `multi-lang-annotated`.

As discussed in the paper's discussion section, we applied the proposed model to conventional commit classification. We collected 50,000 conventional commits from 96 repositories written in 6 languages (C++, Python, Go, Rust, JS, TS). We have made this dataset publicly available for research purposes, and you can access it on Google Drive due to its large size: https://drive.google.com/file/d/1ITytrkS_4R06467Uw1P7VPuA89CtCg7R/view?usp=share_link.

## Dashboard
The source code for the dashboard can be found in the ./dashboard directory, and you can access the demo at https://drive.google.com/file/d/1o-_5hfZEvsINnNGoagFPzM3NvWMMQ8IZ

## Reproduction

The `commit_classifier` folder contains all the code for reproduction.

To reproduce the stratified five-fold cross-validation of our model, use the following command. This command will save results under `reproduction-results/reproduce5fold_multilang`, and the results contain checkpoints, configs, running logs, etc.

``` shell
python launch.py --name reproduce5fold_multilang \
                 --model CCModel \
                 --device gpu \
                 --num_workers 3 \
                 --seed 123 \
                 --data_dir ../dataset/multi-lang-exp/ \
                 --save_dir reproduction-results \
                 --file_num_limit 4 \
                 --hunk_num_limit 3 \
                 --enable_cv \
                 --do_test \
                 --do_train \
                 --enable_checkpoint
```

To reproduce the stratified five-fold cross-validation comparison against baseline on Java programming language, use the following command.

``` shell
python launch.py --name reproduce5fold_java \
                 --model CCModel \
                 --device gpu \
                 --num_workers 3 \
                 --seed 123 \
                 --data_dir ../dataset/1793/ \
                 --save_dir reproduction-results \
                 --file_num_limit 5 \
                 --hunk_num_limit 2 \
                 --enable_cv \
                 --do_test \
                 --do_train \
                 --enable_checkpoint
```

To reproduce the ablation study, use the following command. You can reproduce different variants by using --model arguments, which can be one of `MessageFeatModel`, `MessageCodeModel`, `CodeFeatModel`.

``` shell
python launch.py --name reproduce_ablation \
                 --model MessageCodeModel \
                 --device gpu \
                 --num_workers 3 \
                 --seed 123 \
                 --data_dir ../dataset/multi-lang-exp/ \
                 --save_dir reproduction-results \
                 --file_num_limit 4 \
                 --hunk_num_limit 3 \
                 --enable_checkpoint \
                 --do_train \
                 --do_test
```

Additionally, the `commit_classifier/baselines` folder contains the detailed code for the compared techniques we adopted in the paper, such as ChatGPT.
