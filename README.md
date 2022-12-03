# map-dump

This is a small project aimed at learning "retailer embeddings" using geographic co-occurrences, similar to how [word embeddings](https://en.wikipedia.org/wiki/Word_embedding) leverage word co-occurrences. The goal is to encode every popular retail store as a different vector, such that similar stores are relatively close to each other in the vector space, while unrelated stores are far apart.

For example, we might have some vectors `A`, `B`, and `C` representing `McDonald's`, `Burger King`, and `Urban Outfitters`, respectively. We would like `||A - B|| < ||A - C||`, since `McDonald's` and `Burger King` are both popular fast food chains, whereas Urban Outfitters is a clothing retailer.

# How I did it

Here is a rough sketch of the steps I took:

 1. I scraped an online map service to get millions of `(name, latitude, longitude)` tuples. Notably, this dataset includes landmarks, ATMs, and various other places that aren't strictly stores. In practice, this didn't seem to be a problem, and actually might have helped feature learning.
 2. I computed a geographic "co-occurrence matrix" for the locations. Each row and column corresponds to a different unique location (e.g. "McDonald's"), and entry A_ij is the number of occurrences of i that are within a certain geodesic distance of an occurrence of j. The diagonal of the matrix counts the occurrences of each location. Counts may also be weighted by some function of distance. Note that the matrix is not symmetric.
 3. I wrote a PyTorch implementation of [GloVe](https://nlp.stanford.edu/projects/glove/) to decompose the co-occurrence matrix, resulting in low-dimensional feature vectors for each location.
 4. I tuned co-occurrence and GloVe hyperparameters using performance on a classification task to evaluate the feature space. In particular, I scraped categories for thousands of the locations in the dataset, and trained a classifier to predict the categories from embedding vectors. Here, each data point is a unique location, like McDonald's, not an individual occurrence of a location (like "the McDonald's on 2nd Street"). Thus, the validation set is literally a collection of stores and other landmarks that were not in the training set.
 5. I created a website to browse the resulting embedding space using nearest neighbor lookups.

# Running all the steps

First, you should compile the Rust code with

```
cargo build --release
```

You should also make sure you have Python 3 with PyTorch 1.10 or higher installed for training embeddings. Embedding training will be incredibly slow without a GPU; with a GPU, you will need at least ~10GB of VRAM to fit training.

## Scraping locations

First, we will scrape a coarse level-of-detail map to use to filter the output of the higher level-of-detail scrape.

```bash
./target/release/map_dump discover \
    --min-locations 5 \
    --base-level-of-detail 12 \
    --full-level-of-detail 12 \
    --retries 15 \
    discovered_coarse.json
```

**Download**: [discovered_coarse.json.gz](https://data.aqnichol.com/map-dump/discovered_coarse.json.gz)

Next, we can run a very high-resolution scrape:

```bash
./target/release/map_dump discover \
    --min-locations 5 \
    --base-level-of-detail 12 \
    --full-level-of-detail 14 \
    --retries 15 \
    --filter-scrape discovered_coarse.json \
    discovered_fine.json
```

**Download**: [discovered_fine.json.gz](https://data.aqnichol.com/map-dump/discovered_fine.json.gz)

## Computing co-occurrences

```bash
./target/release/map_dump cooccurrence \
    --min-count 65 \
    --radius .00101040719409922196 \ # 4 miles in radians
    --dropoff-mode InvSquareP10 \
    --sparse-out \
    discovered_fine.json \
    cooccurrence_min65.npz
```

**Download**: [cooccurrence_min65.npz](https://data.aqnichol.com/map-dump/cooccurrence_min65.npz)

## Computing GloVe embeddings

```bash
python3 -u -m glove \
    --iters 20000 \
    --lr 0.01 \
    --weight_decay 0.01 \
    --adam \
    cooccurrence_min65.npz \
    embeddings_min65.json
```

**Download**: [embeddings_min65.json.gz](https://data.aqnichol.com/map-dump/embeddings_min65.json.gz)

## Creating a location index

```bash
./target/release/map_dump location_index \
    discovered_fine.json \
    location_index.zip
```

**Download:** [location_index.zip](https://data.aqnichol.com/map-dump/location_index.zip)

## Scraping categories

```bash
./target/release/map_dump categories \
    --min-count 75 \
    --retries 5 \
    discovered_fine.json \
    categories_min75.json
```

**Download:** [categories_min75.json.gz](https://data.aqnichol.com/map-dump/categories_min75.json.gz)

## Training a classifier

```bash
python3 -u -m classifier \
    --embeddings embeddings_min65.json \
    --categories categories_min75.json \
    --full \
    --classifier linear \
    --trunc 2
```

To see a constant-class baseline, run with `--classifier dummy`.

## Running the website

```
./target/release/map_dump website \
    --location-index location_index.zip \
    Embs embeddings_min65.json
```

The website allows you to pass multiple sets of embeddings, like `name1 path1.json name2 path2.json`, but they must have the exact same store names (in the same order). This will be the default behavior if all of the embeddings are based on the same co-occurrence matrix, or if all of the co-occurrence matrices are based on the same scrape and have the same `--min-count` argument.

# Previous code location

A much older version of this code is located in my [learning-rust](https://github.com/unixpickle/learning-rust/tree/6599848cee5a460925294046a1b9e6986c42d871/map_dump) repo.
