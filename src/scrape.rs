use crate::bing_maps;
use crate::bing_maps::{Client, MapItem};
use crate::geo_coord::GeoBounds;
use crate::task_queue::TaskQueue;
use clap::Parser;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use tokio::fs::create_dir_all;
use tokio::io::AsyncReadExt;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::{
    fs::{remove_file, rename, File},
    io::AsyncWriteExt,
    spawn,
    sync::mpsc::channel,
};

#[derive(Clone, Parser)]
pub struct ScrapeArgs {
    #[clap(short, long, value_parser, default_value_t = 2)]
    max_subdivisions: i32,

    #[clap(short, long, value_parser, default_value_t = 2.0)]
    step_size: f64,

    #[clap(short, long, value_parser, default_value_t = 4)]
    parallelism: u32,

    #[clap(short, long, value_parser, default_value_t = 5)]
    retries: u32,

    #[clap(short, long, value_parser)]
    quiet: bool,

    #[clap(value_parser)]
    names_list: String,

    #[clap(value_parser)]
    output_dir: String,
}

pub async fn scrape(cli: ScrapeArgs) -> anyhow::Result<()> {
    let mut reader = File::open(&cli.names_list).await?;
    let mut contents = String::new();
    reader.read_to_string(&mut contents).await?;
    drop(reader);

    create_dir_all(&cli.output_dir).await?;
    for store_name in contents
        .split("\n")
        .map(|x| x.trim_end())
        .filter(|x| x.len() > 0)
    {
        let out_path = PathBuf::from(&cli.output_dir).join(format!("{}.json", store_name));
        let out_path_str = out_path.to_str().ok_or(anyhow::Error::msg(
            "failed to convert output path to string",
        ))?;
        println!("writing store {} -> {}", store_name, out_path_str);
        scrape_single(&cli, store_name, out_path_str).await?;
        println!("completed store: {}", store_name);
    }
    Ok(())
}

async fn scrape_single(
    cli: &ScrapeArgs,
    store_name: &str,
    output_path: &str,
) -> anyhow::Result<()> {
    let regions = world_regions(cli.step_size);
    let region_count = regions.len().await;
    let (response_tx, response_rx) = channel((cli.parallelism as usize) * 10);
    for _ in 0..cli.parallelism {
        spawn(fetch_regions(
            store_name.to_owned(),
            cli.max_subdivisions,
            cli.retries,
            regions.clone(),
            response_tx.clone(),
        ));
    }
    // Make sure the channel is ended once all the workers finish.
    drop(response_tx);

    let tmp_path = format!("{}.tmp", output_path);
    let mut output = File::create(&tmp_path).await?;
    let mut result = write_outputs(&cli, store_name, &mut output, response_rx, region_count).await;

    // Flush before dropping to ensure the underlying file is closed.
    result = result.and(output.flush().await.map_err(Into::into));
    drop(output);

    if result.is_err() {
        remove_file(tmp_path).await?;
        result
    } else {
        rename(tmp_path, output_path).await?;
        Ok(())
    }
}

async fn write_outputs(
    cli: &ScrapeArgs,
    store_name: &str,
    output: &mut File,
    mut response_rx: Receiver<bing_maps::Result<Vec<MapItem>>>,
    region_count: usize,
) -> anyhow::Result<()> {
    let mut found = HashSet::new();
    let mut completed_regions: usize = 0;
    while let Some(response) = response_rx.recv().await {
        let listing = response?;
        for x in listing {
            if found.insert(x.id.clone()) {
                output
                    .write_all((serde_json::to_string(&x)? + "\n").as_bytes())
                    .await?;
            }
        }
        completed_regions += 1;
        if !cli.quiet
            || completed_regions == region_count
            || completed_regions % (region_count / 100).max(1) == 0
        {
            println!(
                "store \"{}\": completed {}/{} queries ({:.3}%, found {})",
                store_name,
                completed_regions,
                region_count,
                100.0 * (completed_regions as f64) / (region_count as f64),
                found.len()
            );
        }
    }
    Ok(())
}

fn world_regions(step_size: f64) -> TaskQueue<GeoBounds> {
    GeoBounds::globe(step_size).into()
}

async fn fetch_regions(
    store_name: String,
    max_subdivisions: i32,
    max_retries: u32,
    tasks: TaskQueue<GeoBounds>,
    results: Sender<bing_maps::Result<Vec<MapItem>>>,
) {
    let mut client = Client::new();
    while let Some(bounds) = tasks.pop().await {
        let response = fetch_bounds_subdivided(
            &mut client,
            &store_name,
            bounds,
            max_retries,
            max_subdivisions,
        )
        .await;
        let was_ok = response.is_ok();
        if results.send(response).await.is_err() || !was_ok {
            // If we cannot send, it means the main coroutine died
            // due to some error. If we sent an error, there is no
            // point continuing to do work, since the main coroutine
            // will die.
            break;
        }
    }
}

async fn fetch_bounds_subdivided(
    client: &mut Client,
    query: &str,
    bounds: GeoBounds,
    max_retries: u32,
    max_subdivisions: i32,
) -> bing_maps::Result<Vec<MapItem>> {
    // This would be easier with recursion than a depth-first search,
    // but recursion with futures is super annoying and wouldn't allow
    // us to use finite lifetimes for the arguments.
    let initial_results = client.map_search(query, &bounds, max_retries).await?;
    let mut queue = VecDeque::from([(bounds, initial_results, 0)]);
    let mut results = HashMap::new();
    while let Some((bounds, sub_results, depth)) = queue.pop_front() {
        let old_count = results.len();
        for result in sub_results {
            results.insert(result.id.clone(), result);
        }
        let new_count = results.len();

        // Only expand a region if expanding it is still giving new
        // results, indicating that this area is dense with stores.
        if new_count > old_count && depth < max_subdivisions {
            for subdivided in bounds.split() {
                let new_results = client.map_search(query, &subdivided, max_retries).await?;
                queue.push_back((subdivided, new_results, depth + 1));
            }
        }
    }
    Ok(results.into_values().into_iter().collect())
}
