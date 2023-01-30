use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::bing_maps::{Client, Tile};
use crate::discover::fetch_results;
use crate::geo_coord::GeoCoord;
use crate::task_queue::TaskQueue;
use clap::Parser;
use rand::{seq::SliceRandom, thread_rng};
use tokio::task::spawn_blocking;
use tokio::{spawn, sync::mpsc::channel};

#[derive(Clone, Parser)]
pub struct DiscoverAllArgs {
    #[clap(short, long, value_parser, default_value_t = 12)]
    base_level_of_detail: u8,

    #[clap(short, long, value_parser, default_value_t = 14)]
    full_level_of_detail: u8,

    #[clap(short, long, value_parser, default_value_t = 8)]
    parallelism: u32,

    #[clap(short, long, value_parser, default_value_t = 5)]
    retries: u32,

    #[clap(short, long, value_parser, default_value = "*")]
    categories: String,

    #[clap(short, long, value_parser)]
    quiet: bool,

    #[clap(value_parser)]
    output_path: String,
}

pub async fn discover_all(cli: DiscoverAllArgs) -> anyhow::Result<()> {
    let mut all_tiles = Tile::all_tiles(cli.base_level_of_detail);
    let all_tiles_count = all_tiles.len();

    // Restart from previous run.
    let output_path_clone = cli.output_path.clone();
    let seen_tiles = spawn_blocking(move || db_seen_tiles(&output_path_clone)).await??;
    let seen_set = HashSet::<_>::from_iter(seen_tiles.into_iter());
    all_tiles = all_tiles
        .into_iter()
        .filter(|x| !seen_set.contains(x))
        .collect();

    // By shuffling the tiles, we make the progress less bursty, and
    // outputs from the start of the program can be used to predict the
    // duration and final number of results.
    all_tiles.shuffle(&mut thread_rng());
    let queries: TaskQueue<Tile> = all_tiles.into();
    let mut completed = all_tiles_count - queries.len().await;

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
                    .send((
                        tile.clone(),
                        fetch_results(
                            &mut client,
                            &categories,
                            tile,
                            cli.full_level_of_detail,
                            max_retries,
                        )
                        .await,
                    ))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });
    }
    drop(results_tx);

    spawn_blocking(move || -> anyhow::Result<()> {
        let mut db = open_db(&cli.output_path)?;
        while let Some((tile, results)) = results_rx.blocking_recv() {
            let items = results?;
            let tx = db.transaction()?;
            tx.execute(
                "INSERT INTO seen_tiles (lod, x, y) VALUES (?1, ?2, ?3)",
                (tile.level_of_detail as u32, tile.x, tile.y),
            )?;
            for poi in items {
                tx.execute(
                    "INSERT INTO poi (id, name, lat, lon) VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![poi.id, poi.name, poi.location.0, poi.location.1],
                )?;
            }
            tx.commit()?;
            completed += 1;
            if !cli.quiet {
                println!(
                    "completed {}/{} queries ({:.2}%)",
                    completed,
                    all_tiles_count,
                    100.0 * (completed as f64) / (all_tiles_count as f64),
                );
            }
        }
        Ok(())
    })
    .await??;

    Ok(())
}

fn open_db(db_path: &str) -> rusqlite::Result<rusqlite::Connection> {
    let db = rusqlite::Connection::open(db_path)?;
    db.execute(
        "CREATE TABLE if not exists seen_tiles (
            lod INT,
            x   INT,
            y   INT,
            PRIMARY KEY (lod, x, y)
        )",
        (),
    )?;
    db.execute(
        "CREATE TABLE if not exists poi (
            id    TEXT,
            name  TEXT,
            lat   REAL,
            lon   REAL
        )",
        (),
    )?;
    db.execute("CREATE INDEX IF NOT EXISTS poi_name ON poi (name)", ())?;
    Ok(db)
}

fn db_seen_tiles(db_path: &str) -> rusqlite::Result<Vec<Tile>> {
    let conn = open_db(db_path)?;
    let mut stmt = conn.prepare("SELECT * FROM seen_tiles")?;
    let result_it = stmt.query_map((), |row| -> rusqlite::Result<Tile> {
        Ok(Tile {
            level_of_detail: row.get("lod")?,
            x: row.get("x")?,
            y: row.get("y")?,
        })
    })?;
    result_it.into_iter().collect()
}

pub async fn read_all_store_locations_discover_sqlite3<P: 'static + Send + AsRef<Path>>(
    p: P,
    min_count: usize,
) -> anyhow::Result<HashMap<String, Vec<GeoCoord>>> {
    spawn_blocking(move || {
        let db = rusqlite::Connection::open(p)?;
        let mut res = HashMap::<String, Vec<GeoCoord>>::new();
        let mut query = db.prepare(
            "
                SELECT name, lat, lon FROM poi WHERE name IN (
                    SELECT name FROM poi GROUP BY name HAVING COUNT(*) >= ?1
                )
            ",
        )?;
        let results = query.query_map((min_count,), |row| {
            let name = row.get::<_, String>("name")?;
            let lat = row.get::<_, f64>("lat")?;
            let lon = row.get::<_, f64>("lon")?;
            Ok((name, lat, lon))
        })?;
        for item in results {
            let (name, lat, lon) = item?;
            res.entry(name).or_default().push(GeoCoord(lat, lon));
        }
        Ok(res)
    })
    .await?
}
