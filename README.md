# map-dump

This is a small project aimed at learning "retailer embeddings" using geographic co-occurrences, similar to how [word embeddings](https://en.wikipedia.org/wiki/Word_embedding) leverage word co-occurrences. The goal is to encode every popular retail store as a different vector, such that similar stores are relatively close to each other in the vector space, while unrelated stores are far apart.

For example, we might have some vectors `A`, `B`, and `C` representing `McDonald's`, `Burger King`, and `Urban Outfitters`, respectively. We would like `||A - B|| < ||A - C||`, since `McDonald's` and `Burger King` are both popular fast food chains, whereas Urban Outfitters is a clothing retailer.

# How I did it

 1. I started by creating a list of ~100 retailers using a combination of Google, my own memory, and GPT-3. This can be found in [retailers.txt](retailers.txt).
 2. I scraped Bing maps to find every location of each of these stores on the map. The locations are literally stored as `(latitude,longitude)` pairs.
 3. I computed a geographic "co-occurrence matrix" for the stores. Each row and column corresponds to a retailer, and entry A_ij is the number of locations of retailer i that are within a certain geodesic distance of a location of retailer j. The diagonal of the matrix counts the locations of each retailer. Note that the matrix is not symmetric.
 4. I wrote a toy PyTorch implementation of [GloVe](https://nlp.stanford.edu/projects/glove/) to decompose the co-occurrence matrix, resulting in low-dimensional feature vectors for each retailer.
 5. I plotted the embeddings of GloVe using UMAP, and hand-inspected nearest neighbors for some retailers.

# Previous code location

An older version of this code is located in my [learning-rust](https://github.com/unixpickle/learning-rust/tree/6599848cee5a460925294046a1b9e6986c42d871/map_dump) repo.
