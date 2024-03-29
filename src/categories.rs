use crate::bing_maps::Client;
use crate::bing_maps::PointOfInterest;
use crate::cooccurrence::read_all_store_locations_discover;
use crate::geo_coord::GlobeBounds;
use crate::task_queue::TaskQueue;
use clap::Parser;
use rand::seq::IteratorRandom;
use rand::SeedableRng;
use serde::Serialize;
use std::collections::HashMap;
use tokio::{fs::File, io::AsyncWriteExt, spawn, sync::mpsc::channel};

#[derive(Clone, Parser)]
pub struct CategoriesArgs {
    #[clap(short, long, value_parser, default_value_t = 5)]
    sample_locations: usize,

    #[clap(short, long, value_parser, default_value_t = GlobeBounds::Globe)]
    bounds: GlobeBounds,

    #[clap(short, long, value_parser, default_value_t = 8)]
    parallelism: u32,

    #[clap(short, long, value_parser, default_value_t = 5)]
    retries: u32,

    #[clap(short, long, value_parser, default_value_t = 0)]
    min_count: usize,

    #[clap(value_parser)]
    discover_out: String,

    #[clap(value_parser)]
    output_file: String,
}

pub async fn categories(cli: CategoriesArgs) -> anyhow::Result<()> {
    println!("loading locations...");
    let locations = read_all_store_locations_discover(&cli.discover_out, cli.min_count)
        .await?
        .into_iter()
        .map(|(name, locations)| {
            (
                name,
                locations
                    .into_iter()
                    .filter(|x| cli.bounds.contains(&x.location))
                    .collect::<Vec<_>>(),
            )
        })
        .filter(|(_, locations)| locations.len() >= cli.min_count)
        .collect::<HashMap<_, _>>();

    let total_tasks = locations.len();
    let task_queue: TaskQueue<(String, Vec<PointOfInterest>)> = locations.into();
    println!("total locations: {}", total_tasks);

    println!("creating workers...");
    let (result_tx, mut result_rx) = channel(100);

    for i in 0..cli.parallelism {
        let task_queue_clone = task_queue.clone();
        let result_tx_clone = result_tx.clone();
        let max_retries = cli.retries;
        let sample_locations = cli.sample_locations;
        let mut rng = rand::rngs::StdRng::seed_from_u64(i as u64);
        spawn(async move {
            let mut client = Client::new();
            while let Some((name, pois)) = task_queue_clone.pop().await {
                let obj =
                    get_categories(&mut client, &mut rng, pois, sample_locations, max_retries)
                        .await;
                if result_tx_clone.send(obj.map(|x| (name, x))).await.is_err() {
                    break;
                }
            }
        });
    }
    drop(result_tx);

    println!("aggregating outputs...");
    let mut out_map = HashMap::new();
    let mut completed = 0;
    let mut num_empty = 0;
    while let Some(result) = result_rx.recv().await {
        completed += 1;
        match result {
            Err(e) => {
                println!("error fetching info: {}", e);
                num_empty += 1;
            }
            Ok((k, v)) => {
                if v.len() == 0 {
                    num_empty += 1;
                }
                out_map.insert(k, v);
                println!(
                    "completed {}/{} ({:.5}%, {:.5}% not found)",
                    completed,
                    total_tasks,
                    100.0 * (completed as f64) / (total_tasks as f64),
                    100.0 * (num_empty as f64) / (completed as f64)
                );
            }
        }
    }

    println!("writing to {}...", cli.output_file);
    let mut out_file = File::create(cli.output_file).await?;
    out_file.write_all(&serde_json::to_vec(&out_map)?).await?;
    out_file.flush().await?;

    Ok(())
}

#[derive(Serialize)]
struct Category {
    name: Option<String>,
    path: String,
}

async fn get_categories(
    client: &mut Client,
    rng: &mut rand::rngs::StdRng,
    pois: Vec<PointOfInterest>,
    max_count: usize,
    max_retries: u32,
) -> anyhow::Result<Vec<Category>> {
    if pois.len() == 0 {
        return Ok(Vec::new());
    }
    let mut res = Vec::new();
    for poi in pois
        .into_iter()
        .choose_multiple(rng, max_count.min(max_count))
    {
        if let Some(x) = client.id_lookup(&poi.id, max_retries).await? {
            if let Some(path) = x.category_path {
                res.push(Category {
                    name: x.category_name,
                    path,
                });
            }
        }
    }
    Ok(res)
}
