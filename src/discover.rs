use std::collections::{HashMap, HashSet};

use crate::bing_maps::{self, Client, PointOfInterest, Tile};
use crate::task_queue::TaskQueue;
use clap::Parser;
use rand::{seq::SliceRandom, thread_rng};
use tokio::io::AsyncReadExt;
use tokio::{fs::File, io::AsyncWriteExt, spawn, sync::mpsc::channel};

#[derive(Clone, Parser)]
pub struct DiscoverArgs {
    #[clap(short, long, value_parser, default_value_t = 8)]
    base_level_of_detail: u8,

    #[clap(short, long, value_parser, default_value_t = 10)]
    full_level_of_detail: u8,

    #[clap(short, long, value_parser, default_value_t = 8)]
    parallelism: u32,

    #[clap(short, long, value_parser, default_value_t = 5)]
    retries: u32,

    #[clap(short, long, value_parser, default_value = "*")]
    categories: String,

    #[clap(short, long, value_parser, default_value_t = 30)]
    min_locations: u64,

    #[clap(short, long, value_parser)]
    quiet: bool,

    #[clap(long, value_parser)]
    filter_scrape: Option<String>,

    #[clap(value_parser)]
    output_path: String,
}

pub async fn discover(cli: DiscoverArgs) -> anyhow::Result<()> {
    let mut all_tiles = Tile::all_tiles(cli.base_level_of_detail);
    let total_queries = all_tiles.len();

    // By shuffling the tiles, we make the progress less bursty, and
    // outputs from the start of the program can be used to predict the
    // duration and final number of results.
    all_tiles.shuffle(&mut thread_rng());
    let queries: TaskQueue<Tile> = all_tiles.into();

    // A filter may be provided to use only certain store names.
    // This can save memory when scraping very many levels of detail.
    let filter = read_filtered_names(&cli.filter_scrape).await?;

    let (results_tx, mut results_rx) = channel(cli.parallelism as usize);

    for _ in 0..cli.parallelism {
        let queries_clone = queries.clone();
        let results_tx_clone = results_tx.clone();
        let categories = cli
            .categories
            .split(",")
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        let max_retries = cli.retries;
        spawn(async move {
            let mut client = Client::new();
            while let Some(tile) = queries_clone.pop().await {
                if results_tx_clone
                    .send(
                        fetch_results(
                            &mut client,
                            &categories,
                            tile,
                            cli.full_level_of_detail,
                            max_retries,
                        )
                        .await,
                    )
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });
    }
    drop(results_tx);

    let mut results = HashMap::new();
    let mut store_counts = HashMap::new();
    let mut completed = 0;
    while let Some(item) = results_rx.recv().await {
        for result in item? {
            let name = result.name.clone();
            if let Some(f) = &filter {
                if !f.contains(&name.to_lowercase()) {
                    continue;
                }
            }
            if let Some(old) = results.insert(result.id.clone(), result) {
                // It's possible the old entry has a different name than the
                // new one, even though the ID is the same. Perhaps this
                // happens when the store's metadata is updated.
                *store_counts.get_mut(&old.name).unwrap() -= 1;
            }
            *store_counts.entry(name).or_insert(0u64) += 1;
        }
        completed += 1;
        if !cli.quiet {
            println!(
                "completed {}/{} queries ({:.2}%, {} points found)",
                completed,
                total_queries,
                100.0 * (completed as f64) / (total_queries as f64),
                results.len()
            );
        }
    }

    let filtered_locations = results
        .into_values()
        .filter(|x| store_counts[&x.name] >= cli.min_locations)
        .collect::<Vec<_>>();

    println!("filtered to {} points", filtered_locations.len());

    let mut writer = File::create(cli.output_path).await?;
    writer
        .write_all(serde_json::to_string(&filtered_locations)?.as_ref())
        .await?;
    writer.flush().await?;

    Ok(())
}

async fn read_filtered_names(
    path_or_none: &Option<String>,
) -> anyhow::Result<Option<HashSet<String>>> {
    if let Some(x) = path_or_none.as_deref() {
        let mut f = File::open(x).await?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).await?;
        let parsed: Vec<PointOfInterest> = serde_json::from_slice(&buf)?;
        let mut res = HashSet::new();
        for x in parsed {
            res.insert(x.name.to_lowercase());
        }
        Ok(Some(res))
    } else {
        Ok(None)
    }
}

async fn fetch_results(
    client: &mut Client,
    categories: &Vec<String>,
    tile: Tile,
    full_level_of_detail: u8,
    max_retries: u32,
) -> bing_maps::Result<Vec<PointOfInterest>> {
    let mut sub_tiles = Vec::new();
    tile.children_at_lod(full_level_of_detail, &mut sub_tiles);

    let mut res = Vec::new();
    for category in categories {
        let sub_results = client
            .points_of_interest(&tile, category, None, None, max_retries)
            .await?;
        // Only search deeper if some results were found at the base
        // level of detail.
        if sub_results.len() > 0 {
            for child in sub_tiles.iter() {
                res.extend(
                    client
                        .points_of_interest(child, category, None, None, max_retries)
                        .await?,
                );
            }
        }
        res.extend(sub_results);
    }
    Ok(res)
}

pub async fn read_discover_output(
    path: &str,
    min_count: usize,
) -> anyhow::Result<HashMap<String, Vec<PointOfInterest>>> {
    let mut reader = File::open(path).await?;
    let mut contents = String::new();
    reader.read_to_string(&mut contents).await?;
    let locations: Vec<PointOfInterest> = serde_json::from_str(&contents)?;
    let mut res = HashMap::new();
    for location in locations {
        res.entry(location.name.clone())
            .or_insert_with(Vec::new)
            .push(location);
    }
    Ok(res
        .into_iter()
        .filter(|(_name, locations)| locations.len() >= min_count)
        .collect())
}
