use crate::array_util::SparseMatrix;
use crate::bing_maps::{MapItem, PointOfInterest};
use crate::discover::read_all_store_locations_discover_json;
use crate::discover_all::read_all_store_locations_discover_sqlite3;
use crate::geo_coord::{GeoCoord, GlobeBounds, VecGeoCoord};
use crate::npz_file::NpzWriter;
use clap::arg_enum;
use serde_json::{Map, Value};
use std::f64::consts::PI;
use std::io::ErrorKind;
use std::path::Path;
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

const MAX_THREAD_ALLOCATION: usize = 131072;

#[derive(Clone, Parser)]
pub struct CoocurrenceArgs {
    #[clap(value_parser)]
    input_dir: String,

    #[clap(value_parser)]
    output_path: String,

    #[clap(short, long, value_parser, default_value_t = GlobeBounds::Globe)]
    bounds: GlobeBounds,

    // Default radius is roughly 1 mile (in radians).
    #[clap(short, long, value_parser, default_value_t = 0.00025260179852480549)]
    radius: f64,

    #[clap(short, long, value_parser, default_value_t = DropoffMode::Constant)]
    dropoff_mode: DropoffMode,

    #[clap(short, long)]
    count_pairs: bool,

    #[clap(short, long, value_parser, default_value_t = 8)]
    workers: i32,

    #[clap(short, long, value_parser, default_value_t = 0)]
    min_count: usize,

    #[clap(short, long)]
    sparse_out: bool,
}

pub async fn cooccurrence(cli: CoocurrenceArgs) -> anyhow::Result<()> {
    println!("loading locations from: {}", cli.input_dir);
    let input_dir = PathBuf::from(cli.input_dir);
    let store_locations = read_all_store_locations(&input_dir, cli.bounds, cli.min_count).await?;
    println!("loaded {} locations", store_locations.len());

    // Get a canonical ordering of stores for the matrix.
    let mut sorted_names = store_locations
        .keys()
        .map(|x| x.clone())
        .collect::<Vec<_>>();
    sorted_names.sort();

    // Create a flat list of all store locations, sorted by latitude
    // for faster "nearby" lookups.
    let mut flat_locations = Vec::new();
    for (i, name) in sorted_names.iter().enumerate() {
        for location in &store_locations[name] {
            flat_locations.push(Location {
                store_index: i,
                latitude: location.0 * PI / 180.0,
                location: VecGeoCoord::from(location),
            });
        }
    }
    flat_locations.sort_by(|a, b| a.latitude.partial_cmp(&b.latitude).unwrap());

    println!("computing cooccurrence matrix...");

    let cur_index = Arc::new(AtomicUsize::new(0));
    let (results_tx, mut results_rx) = channel(cli.workers as usize);
    for _ in 0..cli.workers {
        let results_tx_clone = results_tx.clone();
        let flat_locations_clone = flat_locations.clone();
        let num_stores = store_locations.len();
        let radius = cli.radius;
        let dropoff_mode = cli.dropoff_mode.clone();
        let count_pairs = cli.count_pairs;
        let cur_index_clone = cur_index.clone();
        spawn_blocking(move || {
            let cos_radius = radius.cos();
            let mut pair_count = SparseMatrix::zeros((num_stores, num_stores));
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
                let src_loc = &flat_locations_clone[src];
                let min_lat = src_loc.latitude - radius;
                let max_lat = src_loc.latitude + radius;
                for (dst, dst_loc) in bisect_locations(&flat_locations_clone, min_lat, max_lat) {
                    let cos_dist = src_loc.location.cos_geo_dist(&dst_loc.location);
                    if cos_dist > cos_radius {
                        let weight = dropoff_mode.weight(cos_dist, radius);
                        bin_row[dst_loc.store_index] = weight.max(bin_row[dst_loc.store_index]);
                        if count_pairs && src > dst {
                            pair_count
                                .add_entry((src_loc.store_index, dst_loc.store_index), weight);
                            pair_count
                                .add_entry((dst_loc.store_index, src_loc.store_index), weight);
                        }
                    }
                }
                for (i, x) in bin_row.iter().enumerate() {
                    binary_count.add_entry((src_loc.store_index, i), *x);
                }
                // Prevent thread-local buffers from becoming too large,
                // since then memory usage will scale with thread count.
                if binary_count.allocated() + pair_count.allocated() > MAX_THREAD_ALLOCATION {
                    results_tx_clone
                        .blocking_send((pair_count.swap_zeros(), binary_count.swap_zeros()))
                        .unwrap();
                }
            }
            results_tx_clone
                .blocking_send((pair_count, binary_count))
                .unwrap();
        });
    }
    // Make sure we don't block on reading results.
    drop(results_tx);

    let mut pair_counts = SparseMatrix::zeros((store_locations.len(), store_locations.len()));
    let mut binary_counts = pair_counts.clone();
    while let Some((pair_count, binary_count)) = results_rx.recv().await {
        pair_counts += &pair_count;
        binary_counts += &binary_count;
    }

    println!("serializing resulting matrix to {}...", cli.output_path);
    let store_counts = sorted_names
        .iter()
        .map(|x| store_locations[x].len() as u64)
        .collect::<Vec<_>>();
    if cli.output_path.ends_with(".json") {
        let mut output_map = Map::from_iter([
            ("radius".to_owned(), Value::from(cli.radius)),
            ("names".to_owned(), Value::from(sorted_names.clone())),
            ("store_counts".to_owned(), store_counts.into()),
            (
                "binary_counts".to_owned(),
                binary_counts.into_json(cli.sparse_out),
            ),
        ]);
        if cli.count_pairs {
            output_map.insert(
                "pair_counts".to_owned(),
                pair_counts.into_json(cli.sparse_out),
            );
        }
        let result_dict = Value::Object(output_map);
        let serialized = serde_json::to_string(&result_dict)?;
        let mut writer = File::create(cli.output_path).await?;
        writer.write_all(serialized.as_bytes()).await?;
        writer.flush().await?;
    } else {
        let mut writer = NpzWriter::new(&cli.output_path).await?;
        writer.write("radius", cli.radius).await?;
        writer.write("names", sorted_names.clone()).await?;
        writer.write("store_counts", store_counts).await?;
        binary_counts
            .write_to_npz(&mut writer, "binary_counts", cli.sparse_out)
            .await?;
        if cli.count_pairs {
            pair_counts
                .write_to_npz(&mut writer, "pair_counts", cli.sparse_out)
                .await?;
        }
        writer.close().await?;
    }

    Ok(())
}

pub async fn read_all_store_locations<P: AsRef<Path>>(
    src: P,
    bounds: GlobeBounds,
    min_count: usize,
) -> anyhow::Result<HashMap<String, Vec<GeoCoord>>> {
    let all_results = if metadata(&src).await?.is_dir() {
        read_all_store_locations_scrape(&src.as_ref().to_owned()).await?
    } else {
        read_all_store_locations_discover(&src.as_ref().to_owned(), min_count)
            .await?
            .into_iter()
            .map(|(name, pois)| (name, pois.into_iter().map(|poi| poi.location).collect()))
            .collect()
    };
    Ok(all_results
        .into_iter()
        .map(|(name, locations)| {
            (
                name,
                locations
                    .into_iter()
                    .filter(|x| bounds.contains(x))
                    .collect::<Vec<GeoCoord>>(),
            )
        })
        .filter(|(_, locations)| locations.len() >= min_count)
        .collect())
}

pub async fn read_all_store_locations_discover<P: AsRef<Path>>(
    input_path: P,
    min_count: usize,
) -> anyhow::Result<HashMap<String, Vec<PointOfInterest>>> {
    if is_sqlite3_file(input_path.as_ref()).await? {
        read_all_store_locations_discover_sqlite3(input_path.as_ref().to_owned(), min_count).await
    } else {
        read_all_store_locations_discover_json(input_path, min_count).await
    }
}

async fn is_sqlite3_file(input_path: &Path) -> std::io::Result<bool> {
    let mut contents: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    match File::open(input_path)
        .await?
        .read_exact(&mut contents)
        .await
    {
        Ok(_) => Ok(&contents == b"SQLite format 3\0"),
        Err(e) => {
            if e.kind() == ErrorKind::UnexpectedEof {
                Ok(false)
            } else {
                Err(e)
            }
        }
    }
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

arg_enum! {
    #[derive(Clone)]
    enum DropoffMode {
        Constant,
        Linear,
        Quadratic,
        InvSquareP50,
        InvSquareP25,
        InvSquareP10,
    }
}

impl DropoffMode {
    fn weight(&self, cos_dist: f64, radius: f64) -> f64 {
        match self {
            DropoffMode::Constant => 1.0,
            DropoffMode::Linear => 1.0 - (cos_dist.clamp(-1.0, 1.0).acos() / radius),
            DropoffMode::Quadratic => 1.0 - (cos_dist.clamp(-1.0, 1.0).acos() / radius).powf(2.0),
            DropoffMode::InvSquareP50 | DropoffMode::InvSquareP25 | DropoffMode::InvSquareP10 => {
                let theta = cos_dist.clamp(-1.0, 1.0).acos();
                let percentile = match self {
                    DropoffMode::InvSquareP50 => 0.5,
                    DropoffMode::InvSquareP25 => 0.25,
                    DropoffMode::InvSquareP10 => 0.1,
                    _ => unreachable!(),
                };
                let min_radius = radius * percentile;
                (min_radius / theta.max(min_radius)).powf(2.0)
            }
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
struct Location {
    store_index: usize,
    latitude: f64, // radians
    location: VecGeoCoord,
}

fn bisect_locations<'a>(
    locs: &'a Vec<Location>,
    min: f64,
    max: f64,
) -> impl Iterator<Item = (usize, &Location)> + 'a {
    let start = locs.partition_point(move |x| x.latitude < min);
    locs[start..locs.len()]
        .iter()
        .enumerate()
        .map(move |(i, x)| (i + start, x))
        .take_while(move |(_, x)| x.latitude <= max)
}

#[cfg(test)]
mod tests {
    use super::{bisect_locations, GeoCoord, Location, VecGeoCoord};

    #[test]
    fn test_bisect_locations() {
        let locations = [-1.0, -0.99, -0.85, -0.3, 0.1, 0.5, 0.8, 0.95, 0.99]
            .into_iter()
            .map(|x| Location {
                store_index: 0,
                location: VecGeoCoord::from(&GeoCoord(x, 0.0)),
                latitude: x,
            })
            .collect();
        let cases = [
            (-1.1, -1.0),
            (-1.1, -0.995),
            (-1.0, -0.99),
            (-1.01, -0.98),
            (-0.98, -0.5),
            (-0.98, 1.0),
            (0.995, 1.0),
            (-0.2, 1.0),
            (-0.2, 0.9),
        ];
        for (min, max) in cases {
            let actual = collect_locations(bisect_locations(&locations, min, max));
            let expected = collect_locations(bisect_locations_dummy(&locations, min, max));
            assert_eq!(actual, expected, "bad results for case {},{}", min, max);
        }
    }

    fn bisect_locations_dummy<'a>(
        locs: &'a Vec<Location>,
        min: f64,
        max: f64,
    ) -> impl Iterator<Item = (usize, &Location)> + 'a {
        locs.iter()
            .enumerate()
            .skip_while(move |(_, x)| x.latitude < min)
            .take_while(move |(_, x)| x.latitude <= max)
    }

    fn collect_locations<'a, I: Iterator<Item = (usize, &'a Location)>>(
        x: I,
    ) -> Vec<(usize, Location)> {
        x.map(|(i, loc)| (i, loc.clone())).collect()
    }
}
