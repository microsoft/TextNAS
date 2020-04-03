This is the implementation of the TextNAS algorithm proposed in the paper [TextNAS: A Neural Architecture Search Space tailored for Text Representation](https://arxiv.org/pdf/1912.10729.pdf). TextNAS is a neural architecture search algorithm tailored for text representation, more specifically, TextNAS is based on a novel search space consists of operators widely adopted to solve various NLP tasks, and TextNAS also supports multi-path ensemble within a single network to balance the width and depth of the architecture. 

The search space of TextNAS contains: 

    * 1-D convolutional operator with filter size 1, 3, 5, 7 
    * recurrent operator (bi-directional GRU) 
    * self-attention operator
    * pooling operator (max/average)

Following the ENAS algorithm, TextNAS also utilizes parameter sharing to accelerate the search speed and adopts a reinforcement-learning controller for the architecture sampling and generation. Please refer to the paper for more details of TextNAS.

## Preparation

Prepare the word vectors and SST dataset, and organize them in data directory as shown below:

```
textnas
├── data
│   ├── sst
│   │   ├── dev.txt
│   │   ├── test.txt
│   │   └── train.txt
│   └── glove.840B.300d.txt
├── dataloader.py
├── model.py
├── ops.py
├── README.md
├── search.py
└── utils.py
```

The following link might be helpful for finding and downloading the corresponding dataset:

* [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
  * [glove.840B.300d.txt](http://nlp.stanford.edu/data/glove.840B.300d.zip)
* [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://nlp.stanford.edu/sentiment/)
  * [trainDevTestTrees_PTB.zip](https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip)

## Examples

### Search Space

```bash
# If the code is cloned already, ignore this line and enter code folder.
git clone https://github.com/microsoft/TextNAS.git

# search the best architecture
cd TextNAS

# view more options for search
sh sst_macro_search_multi.sh
```
### Evaluate the architecture

```bash
# If the code is cloned already, ignore this line and enter code folder.
git clone https://github.com/microsoft/TextNAS.git

# evaluate the architecture on sst-2
sh eval_arc_sst2.sh

# evaluate the architecture on sst-5
```
sh eval_arc_sst5.sh

## Citations

If you happen to use our work, please consider citing our paper.

@article{wang2019textnas,
  title={TextNAS: A Neural Architecture Search Space tailored for Text Representation},
  author={Wang, Yujing and Yang, Yaming and Chen, Yiren and Bai, Jing and Zhang, Ce and Su, Guinan and Kou, Xiaoyu and Tong, Yunhai and Yang, Mao and Zhou, Lidong},
  journal={arXiv preprint arXiv:1912.10729},
  year={2019}
}
