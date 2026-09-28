# Third-party models and corpus

## ColBERT

[Jina AI's jina-colbert-v2](https://huggingface.co/jinaai/jina-colbert-v2) declares **CC BY-NC 4.0**. Retain attribution to Jina AI and the model card. Commercial deployment needs an appropriate licensing basis; this repository grants no additional model rights. Consult the [license terms](https://creativecommons.org/licenses/by-nc/4.0/) and model owner for commercial licensing.

Its FastEmbed ONNX artifact is approximately 2.2 GB. A local compatibility workaround can copy the ONNX files, increasing cache size. Weights download at runtime and are not included in the repository.

## Other encoders and content

- Sparse encoder: [Qdrant/bm25](https://huggingface.co/Qdrant/bm25), configured for Portuguese.
- Dense encoder: [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small).
- Corpus: Portuguese [GitHub documentation](https://docs.github.com/pt). Each chunk retains source URL, title, section and source ID. See [the frozen corpus](../data/corpus/chunks.jsonl) and [preparation notes](corpus-plan.md).

Follow upstream licenses and notices. The corpus is third-party documentation, not original project prose. See [GitHub Docs licensing information](https://github.com/github/docs/blob/main/LICENSE).

## Repository code

The original code in this repository is licensed under the [MIT License](../LICENSE). That license covers this repository code only and does not relicense the models, their weights, or the GitHub Docs corpus. The project is a portfolio and learning project; any future use of third-party artifacts must still follow their own terms.
