use crate::{
    cooccurrence::read_all_store_locations,
    geo_coord::{GeoCoord, GlobeBounds},
};
use clap::Parser;
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Write},
    path::Path,
};
use tokio::{
    sync::{
        mpsc::{channel, Sender},
        oneshot,
    },
    task::spawn_blocking,
};
use zip::{write::FileOptions, ZipArchive, ZipWriter};

#[derive(Clone, Parser)]
pub struct LocationIndexArgs {
    #[clap(short, long, value_parser, default_value_t = 1)]
    min_count: usize,

    #[clap(short, long, value_parser, default_value_t = GlobeBounds::Globe)]
    bounds: GlobeBounds,

    #[clap(value_parser)]
    discover_out: String,

    #[clap(value_parser)]
    output_file: String,
}

pub async fn location_index(cli: LocationIndexArgs) -> anyhow::Result<()> {
    println!("loading locations...");
    let locations = read_all_store_locations(&cli.discover_out, cli.bounds, cli.min_count).await?;
    let keys = locations.keys().map(|x| x.clone()).collect::<Vec<_>>();

    println!("writing data to file...");
    spawn_blocking(move || -> anyhow::Result<()> {
        let file = File::create(&cli.output_file).unwrap();
        let mut zw = ZipWriter::new(file);

        let name_json = serde_json::to_vec(&keys)?;
        zw.start_file("names.json", FileOptions::default())?;
        zw.write_all(&name_json)?;

        for (i, k) in keys.into_iter().enumerate() {
            let coords = locations[&k].iter().map(|x| x.clone()).collect::<Vec<_>>();
            let encoded = serde_json::to_vec(&coords)?;
            zw.start_file(format!("{}.json", i), FileOptions::default())?;
            zw.write_all(&encoded)?;
        }
        zw.finish()?;
        Ok(())
    })
    .await??;
    Ok(())
}

pub struct LocationIndex {
    comm: Sender<(
        String,
        oneshot::Sender<anyhow::Result<Option<Vec<GeoCoord>>>>,
    )>,
}

impl LocationIndex {
    pub async fn new<P: AsRef<Path>>(path: P) -> anyhow::Result<LocationIndex> {
        let path_clone = path.as_ref().to_owned();
        spawn_blocking(move || -> anyhow::Result<LocationIndex> {
            LocationIndex::new_blocking(&path_clone)
        })
        .await?
    }

    fn new_blocking(path: &Path) -> anyhow::Result<LocationIndex> {
        let file = File::open(path)?;
        let mut zf = ZipArchive::new(file)?;
        let mut f = zf.by_name("names.json")?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        drop(f);
        let names: Vec<String> = serde_json::from_slice(&buf)?;
        let name_to_index = names
            .into_iter()
            .enumerate()
            .map(|(i, x)| (x, i))
            .collect::<HashMap<_, _>>();

        let (tx, mut rx) = channel::<(
            String,
            oneshot::Sender<anyhow::Result<Option<Vec<GeoCoord>>>>,
        )>(100);
        spawn_blocking(move || {
            while let Some((name, response_tx)) = rx.blocking_recv() {
                let idx = name_to_index.get(&name).map(|x| *x);
                let response = LocationIndex::read_locations(&mut zf, idx);
                response_tx.send(response).ok();
            }
        });

        Ok(LocationIndex { comm: tx })
    }

    pub async fn lookup(&self, name: &str) -> anyhow::Result<Option<Vec<GeoCoord>>> {
        let (tx, rx) = oneshot::channel();
        self.comm.send((name.to_owned(), tx)).await?;
        rx.await?
    }

    fn read_locations<R: std::io::Read + std::io::Seek>(
        zf: &mut ZipArchive<R>,
        idx: Option<usize>,
    ) -> anyhow::Result<Option<Vec<GeoCoord>>> {
        match idx {
            None => Ok(None),
            Some(idx) => {
                let mut f = zf.by_name(&format!("{}.json", idx))?;
                let mut buf = Vec::new();
                f.read_to_end(&mut buf)?;
                Ok(Some(serde_json::from_slice(&buf)?))
            }
        }
    }
}
