# CEQE: Contextualized Embeddings for Query Expansion
This repo contains the code for our paper [CEQE: Contextualized Embeddings for Query Expansion](https://arxiv.org/pdf/2103.05256.pdf).

### Introduction

CEQE (Contextualized Embeddings for Query Expansion) is a query expansion model that leverage pre-trained language models and relevance model principles to rank expansion terms for a given query. We support BERT as the pre-trained language model.

### Getting Started

To run CEQE, you need to have the following data:
- A TREC format initial run file. (example: [robust_bm25.combined.run](https://github.com/sherinaseri/ceqe-release/tree/main/data/robust))
- The text of the top k retrieved documents in the initial run. (example of the 1000 retireved documents for query_id 301: [prfdocs.bm25/301](https://github.com/sherinaseri/ceqe-release/tree/main/data/robust/prfdocs.bm25/301))
- The query id and query text in a tab separated file. (example: [stopped_queries_lower.txt](https://github.com/sherinaseri/ceqe-release/blob/main/data/robust/stopped_queries_lower.txt))

To rank the expansion term for a given query (for example query "301" in Robust04 collection) run `end-to-end-ranking-expansion-terms.py`:
```
python end-to-end-ranking-expansion-terms.py --query_id="301"  
    --output_dir=output_dir 
    --query_file=./data/robust/stopped_queries_lower.txt  
    --prf_docs_path=./data/robust/prfdocs.bm25/ 
    --run_file=./data/robust/robust_bm25.combined.run
```

### Environment
* Install [Huggingface Transformers](https://github.com/huggingface/transformers)
* Developed with Python 3.8, Torch 1.7.0, and Transformers 2.3.0

### Citation
If you find this paper/code useful, please cite:
```
@inproceedings{naseri2021ceqe,
  author    = {Shahrzad Naseri and
               Jeff Dalton and
               Andrew Yates and
               James Allan},
  title     = {{CEQE:} Contextualized Embeddings for Query Expansion},
  booktitle = {Advances in Information Retrieval - 43rd European Conference on {IR}
               Research, {ECIR} 2021, Virtual Event, March 28 - April 1, 2021, Proceedings,
               Part {I}},
  volume    = {12656},
  pages     = {467--482},
  publisher = {Springer},
  year      = {2021}
}
```
