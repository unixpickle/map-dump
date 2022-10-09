use clap::Parser;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server};
use ndarray::Array2;
use serde::Deserialize;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::{collections::HashMap, sync::Arc};
use tokio::{fs::File, io::AsyncReadExt};

static HOMEPAGE: &str = include_str!("web_assets/index.html");

#[derive(Clone, Parser)]
pub struct WebsiteArgs {
    #[clap(short, long, value_parser, default_value_t = 8080)]
    port: u16,

    #[clap(value_parser)]
    embedding_names_and_paths: Vec<String>,
}

pub async fn website(cli: WebsiteArgs) -> anyhow::Result<()> {
    println!("reading embeddings...");
    let embeddings = Arc::new(Embeddings::read(&cli.embedding_names_and_paths).await?);

    let addr = SocketAddr::from(([0, 0, 0, 0], cli.port));

    let make_service = make_service_fn(move |_conn| {
        let emb_clone = embeddings.clone();
        async move {
            Ok::<_, Infallible>(service_fn(move |req: Request<Body>| {
                handle_request(req, emb_clone.clone())
            }))
        }
    });

    Server::bind(&addr).serve(make_service).await?;

    Ok(())
}

async fn handle_request(
    req: Request<Body>,
    emb: Arc<Embeddings>,
) -> Result<Response<Body>, Infallible> {
    Ok(Response::new(Body::from(HOMEPAGE)))
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct RawEmbeddings {
    vecs: Vec<Vec<f32>>,
    vecs_bias: Vec<f32>,
    contexts: Vec<Vec<f32>>,
    contexts_bias: Vec<f32>,
    store_names: Vec<String>,
    store_counts: Vec<u64>,
}

#[derive(Clone)]
struct Embeddings {
    vecs: HashMap<String, Array2<f32>>,
    name_to_index: HashMap<String, usize>,
    store_names: Vec<String>,
    store_counts: Vec<u64>,
}

impl Embeddings {
    async fn read(names_and_paths: &[String]) -> anyhow::Result<Self> {
        if {
            let x = names_and_paths.len();
            x == 0 || x % 2 != 0
        } {
            return Err(anyhow::Error::msg(
                "expected even number of arguments, in the form of <name> <path>",
            ));
        }
        let mut all_raw: HashMap<String, RawEmbeddings> = HashMap::new();
        for i in (0..names_and_paths.len()).step_by(2) {
            let mut f = File::open(&names_and_paths[i + 1]).await?;
            let mut data = Vec::new();
            f.read_to_end(&mut data).await?;
            all_raw.insert(
                names_and_paths[i].to_owned(),
                serde_json::from_slice(&data)?,
            );
        }
        let first = all_raw.values().next().unwrap();
        for x in all_raw.values() {
            if x.store_names != first.store_names {
                return Err(anyhow::Error::msg(
                    "all embeddings must have exactly the same store names",
                ));
            }
        }
        let mut vecs = HashMap::new();
        for (name, obj) in all_raw.iter() {
            vecs.insert(name.clone(), vec_to_array2(&obj.vecs));
        }
        let mut name_to_index = HashMap::new();
        for (i, name) in first.store_names.iter().enumerate() {
            name_to_index.insert(name.clone(), i);
        }
        Ok(Self {
            vecs: vecs,
            name_to_index: name_to_index,
            store_names: first.store_names.clone(),
            store_counts: first.store_counts.clone(),
        })
    }
}

fn vec_to_array2(v: &Vec<Vec<f32>>) -> Array2<f32> {
    let mut res = Array2::<f32>::zeros((v.len(), v.len()));
    for (i, xs) in v.iter().enumerate() {
        for (j, x) in xs.iter().enumerate() {
            *res.get_mut((i, j)).unwrap() = *x;
        }
    }
    res
}
