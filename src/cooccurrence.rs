use crate::array_util::matrix_to_json;
use crate::bing_maps::MapItem;
use crate::bing_maps::PointOfInterest;
use crate::geo_coord::{GeoCoord, VecGeoCoord};
use ndarray::Array2;
use serde_json::{Map, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tokio::fs::metadata;

use clap::Parser;
use tokio::{
    fs::{read_dir, File},
    io::{AsyncReadExt, AsyncWriteExt},
    sync::mpsc::channel,
    task::spawn_blocking,
};

#[derive(Clone, Parser)]
pub struct CoocurrenceArgs {
    #[clap(value_parser)]
    input_dir: String,

    #[clap(value_parser)]
    output_path: String,

    // Default radius is roughly 1 mile (in radians).
    #[clap(short, long, value_parser, default_value_t = 0.00025260179852480549)]
    radius: f64,

    #[clap(short, long, value_parser, default_value_t = 8)]
    workers: i32,

    #[clap(short, long, value_parser, default_value_t = 0)]
    min_count: usize,

    #[clap(short, long)]
    sparse_out: bool,
}

pub async fn cooccurrence(cli: CoocurrenceArgs) -> anyhow::Result<()> {
    let input_dir = PathBuf::from(cli.input_dir);
    let store_locations = read_all_store_locations(&input_dir, cli.min_count).await?;

    println!("loaded {} locations", store_locations.len());

    // Get a canonical ordering of stores for the matrix.
    let mut sorted_names = store_locations
        .keys()
        .map(|x| x.clone())
        .collect::<Vec<_>>();
    sorted_names.sort();

    // Create a flat list of (store_index, latitude, location) pairs.
    let mut flat_locations = Vec::new();
    for (i, name) in sorted_names.iter().enumerate() {
        for location in &store_locations[name] {
            flat_locations.push((i, location.0, VecGeoCoord::from(location)));
        }
    }
    flat_locations.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("computing cooccurrence matrix...");

    let cur_index = Arc::new(AtomicUsize::new(0));
    let (results_tx, mut results_rx) = channel(cli.workers as usize);
    for _ in 0..cli.workers {
        let results_tx_clone = results_tx.clone();
        let flat_locations_clone = flat_locations.clone();
        let num_stores = store_locations.len();
        let radius = cli.radius;
        let cur_index_clone = cur_index.clone();
        spawn_blocking(move || {
            let cos_radius = radius.cos();
            let mut pair_count = HashMap::<(usize, usize), f32>::new();
            let mut binary_count = pair_count.clone();
            loop {
                let src = cur_index_clone.fetch_add(1, Ordering::SeqCst);
                if src % (flat_locations_clone.len() / 100) == 0 {
                    eprintln!(
                        "done {}/{} ({:.2}%)",
                        src,
                        flat_locations_clone.len(),
                        100.0 * (src as f64) / (flat_locations_clone.len() as f64)
                    );
                }
                if src >= flat_locations_clone.len() {
                    break;
                }
                let mut bin_row = vec![0.0; num_stores];
                let (src_store, src_lat, src_loc) = &flat_locations_clone[src];
                let min_lat = src_lat - radius;
                let max_lat = src_lat + radius;
                for (dst, (dst_store, _, dst_loc)) in
                    bisect_locations(&flat_locations_clone, min_lat, max_lat)
                {
                    if src_loc.cos_geo_dist(dst_loc) > cos_radius {
                        bin_row[*dst_store] = 1.0;
                        if src > dst {
                            *pair_count.entry((*src_store, *dst_store)).or_default() += 1.0;
                            *pair_count.entry((*dst_store, *src_store)).or_default() += 1.0;
                        }
                    }
                }
                for (i, x) in bin_row.iter().enumerate() {
                    if *x > 0.0 {
                        *binary_count.entry((*src_store, i)).or_default() += *x;
                    }
                }
            }
            results_tx_clone
                .blocking_send((pair_count, binary_count))
                .unwrap();
        });
    }
    // Make sure we don't block on reading results.
    drop(results_tx);

    let mut pair_counts = Array2::<f32>::zeros((store_locations.len(), store_locations.len()));
    let mut binary_counts = pair_counts.clone();
    while let Some((pair_count, binary_count)) = results_rx.recv().await {
        for (idx, count) in pair_count.into_iter() {
            pair_counts[idx] += count;
        }
        for (idx, count) in binary_count.into_iter() {
            binary_counts[idx] += count;
        }
    }

    println!("serializing resulting matrix to {}...", cli.output_path);
    let result_dict = Value::Object(Map::from_iter([
        ("radius".to_owned(), Value::from(cli.radius)),
        ("names".to_owned(), Value::from(sorted_names.clone())),
        (
            "store_counts".to_owned(),
            sorted_names
                .iter()
                .map(|x| store_locations[x].len())
                .collect::<Vec<_>>()
                .into(),
        ),
        (
            "pair_counts".to_owned(),
            matrix_to_json(&pair_counts, cli.sparse_out),
        ),
        (
            "binary_counts".to_owned(),
            matrix_to_json(&binary_counts, cli.sparse_out),
        ),
    ]));
    let serialized = serde_json::to_string(&result_dict)?;
    let mut writer = File::create(cli.output_path).await?;
    writer.write_all(serialized.as_bytes()).await?;
    writer.flush().await?;

    Ok(())
}

async fn read_all_store_locations(
    src: &PathBuf,
    min_count: usize,
) -> anyhow::Result<HashMap<String, Vec<GeoCoord>>> {
    let all_results = if metadata(src).await?.is_dir() {
        read_all_store_locations_scrape(src).await?
    } else {
        read_all_store_locations_discover(src).await?
    };
    Ok(all_results
        .into_iter()
        .filter(|(_, locations)| locations.len() >= min_count)
        .collect())
}

async fn read_all_store_locations_discover(
    input_dir: &PathBuf,
) -> anyhow::Result<HashMap<String, Vec<GeoCoord>>> {
    let mut contents = Vec::new();
    File::open(input_dir)
        .await?
        .read_to_end(&mut contents)
        .await?;
    let pois: Vec<PointOfInterest> = serde_json::from_slice(&contents)?;
    let mut results = HashMap::new();
    for poi in pois.into_iter() {
        results
            .entry(poi.name.clone())
            .or_insert_with(Vec::new)
            .push(poi.location);
    }
    Ok(results)
}

async fn read_all_store_locations_scrape(
    input_dir: &PathBuf,
) -> anyhow::Result<HashMap<String, Vec<GeoCoord>>> {
    let mut store_locations = HashMap::new();
    let mut reader = read_dir(&input_dir).await?;
    while let Some(entry) = reader.next_entry().await? {
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::Error::msg("failed to convert strings"))?;
        if name.starts_with(".") {
            continue;
        }
        if let Some(store_name) = name.strip_suffix(".json") {
            store_locations.insert(
                store_name.to_owned(),
                read_scraped_locations(&entry.path()).await?,
            );
        }
    }
    Ok(store_locations)
}

async fn read_scraped_locations(src: &PathBuf) -> anyhow::Result<Vec<GeoCoord>> {
    let mut reader = File::open(src).await?;
    let mut data = String::new();
    reader.read_to_string(&mut data).await?;
    let result = data
        .split("\n")
        .into_iter()
        .filter(|x| x.len() > 0)
        .map(|x| serde_json::from_str::<MapItem>(&x).map(|x| x.location))
        .collect::<Result<Vec<_>, _>>();
    Ok(result?)
}

fn bisect_locations<'a, A, B>(
    locs: &'a Vec<(A, f64, B)>,
    min: f64,
    max: f64,
) -> impl Iterator<Item = (usize, &(A, f64, B))> + 'a {
    let mut start = 0;
    let mut end = locs.len();
    while start < end {
        let mid = (start + end) / 2;
        let val = locs[mid].1;
        if val < min {
            start = mid;
        } else {
            end = mid;
        }
    }
    locs[start..locs.len()]
        .iter()
        .enumerate()
        .map(move |(i, x)| (i + start, x))
        .take_while(move |(_, (_, x, _))| *x <= max)
}
