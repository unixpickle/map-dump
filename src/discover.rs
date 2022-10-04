use std::collections::HashMap;
use std::sync::Arc;

use crate::bing_maps::{self, Client, PointOfInterest, Tile};
use clap::Parser;
use tokio::sync::Mutex;
use tokio::{fs::File, io::AsyncWriteExt, spawn, sync::mpsc::channel};

#[derive(Clone, Parser)]
pub struct DiscoverArgs {
    #[clap(short, long, value_parser, default_value_t = 3)]
    level_of_detail: u8,

    #[clap(short, long, value_parser, default_value_t = 8)]
    parallelism: i32,

    #[clap(short, long, value_parser, default_value_t = 5)]
    retries: u32,

    // Default categories taken from here:
    // https://github.com/morgangrobin/mcgilldemo/tree/eb268215143b54d3be628c959dab90de9765ff23#exercise-7-customizing-the-api-for-personalized-results
    #[clap(
        short,
        long,
        value_parser,
        default_value = "90001,90012,90016,90111,90232,90243,90265,90287,90353,90408,90551,90617,90619,90661,90727,90738,90771,90793,90870,90932,90942,91457,91493,91510,91567"
    )]
    categories: String,

    #[clap(short, long, value_parser, default_value_t = 30)]
    min_locations: u32,

    #[clap(short, long, value_parser)]
    quiet: bool,

    #[clap(value_parser)]
    output_path: String,
}

pub async fn discover(cli: DiscoverArgs) -> anyhow::Result<()> {
    let all_tiles = Tile::all_tiles(cli.level_of_detail);
    let total_queries = all_tiles.len();
    let queries = Arc::new(Mutex::new(all_tiles));

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
            let client = Client::new();
            while let Some(tile) = pop_task(&queries_clone).await {
                if results_tx_clone
                    .send(fetch_results(&client, &categories, tile, max_retries).await)
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
            if results.insert(result.id.clone(), result).is_none() {
                *store_counts.entry(name).or_insert(0) += 1;
            }
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

async fn fetch_results(
    client: &Client,
    categories: &Vec<String>,
    tile: Tile,
    max_retries: u32,
) -> bing_maps::Result<Vec<PointOfInterest>> {
    let mut res = Vec::new();
    for category in categories {
        let sub_results = client
            .points_of_interest(&tile, category, None, None, max_retries)
            .await?;
        res.extend(sub_results);
    }
    Ok(res)
}

async fn pop_task(tasks: &Arc<Mutex<Vec<Tile>>>) -> Option<Tile> {
    tasks.lock().await.pop()
}
