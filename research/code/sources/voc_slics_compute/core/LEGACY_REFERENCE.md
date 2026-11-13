# Native SLiCS reference and task adaptation

Zhi Li, Hau Phan, Matthew Emigh, Austin J. Brockmeier.
Disentangling Latent Embeddings with Sparse Linear Concept Subspaces (SLiCS).
arXiv:2508.20322v1,2025.
https://arxiv.org/html/2508.20322v1

Implementation follows Algorithm1, Eq.2, Eq.3, Eq.5, and Sections3.3--3.3.2.
Mini-batch training and validation-based stopping/checkpoint selection are
explicitly discussed in Section3.3. Its original experiments use whole-image
embeddings and image-level concepts; this study deliberately applies the
published objective and updates to common labelled DINOv3 patch vectors.
It is a same-task methodological comparison, not a reproduction of their
reported retrieval datasets or numbers. All baseline code is included and
tested; it is our paper-based implementation, not a claim of author-certified
or released-checkpoint reproduction.

Patch presence labels are provided by the same COCO masks available to GGF.
GGF uses instance contrast information in its own loss; SLiCS uses supported
coefficient constraints. No GGF object-contrast or detection-head objective is
silently grafted onto SLiCS. Source/code attribution for the reused BSF/GGF
implementation remains in LICENSE and UPSTREAM.json.
