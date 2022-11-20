use crate::array_util::{dense_matrix_to_json, normalize_rows, vec_to_matrix};
use crate::geo_coord::GeoCoord;
use crate::location_index::LocationIndex;
use clap::Parser;
use flate2::write::GzEncoder;
use flate2::Compression;
use futures_util::StreamExt;
use http::StatusCode;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::cmp::{Ord, Ordering};
use std::convert::Infallible;
use std::fmt::Display;
use std::io::Write;
use std::net::SocketAddr;
use std::ops::Deref;
use std::{collections::HashMap, sync::Arc};
use tokio::task::spawn_blocking;
use tokio::{fs::File, io::AsyncReadExt};

static ASSETS: [(&str, &str); 3] = [
    ("/", include_str!("web_assets/index.html")),
    ("/script.js", include_str!("web_assets/script.js")),
    ("/style.css", include_str!("web_assets/style.css")),
];

static NOT_FOUND_PAGE: &str = include_str!("web_assets/404.html");
static WORLD_MAP_JPEG: &[u8] = include_bytes!("web_assets/world_map.jpg");

#[derive(Clone, Parser)]
pub struct WebsiteArgs {
    #[clap(short, long, value_parser, default_value_t = 8080)]
    port: u16,

    #[clap(short, long, value_parser)]
    location_index: String,

    #[clap(value_parser)]
    embedding_names_and_paths: Vec<String>,
}

#[derive(Clone)]
struct ServerState {
    emb: Arc<Embeddings>,
    loc_index: Arc<LocationIndex>,
}

pub async fn website(cli: WebsiteArgs) -> anyhow::Result<()> {
    println!("reading embeddings...");
    let embeddings = Arc::new(Embeddings::read(&cli.embedding_names_and_paths).await?);

    println!("reading location index...");
    let loc_index = Arc::new(LocationIndex::new(&cli.location_index).await?);

    let state = ServerState {
        emb: embeddings,
        loc_index,
    };

    let addr = SocketAddr::from(([0, 0, 0, 0], cli.port));

    let make_service = make_service_fn(move |_conn| {
        let state_clone = state.clone();
        async move {
            Ok::<_, Infallible>(service_fn(move |req: Request<Body>| {
                let state_clone_clone = state_clone.clone();
                async move {
                    let raw_resp = handle_request(&req, state_clone_clone).await.unwrap();
                    maybe_compress_response(&req, raw_resp).await
                }
            }))
        }
    });

    println!("creating server at {}...", addr);
    Server::bind(&addr).serve(make_service).await?;

    Ok(())
}

async fn handle_request(
    req: &Request<Body>,
    state: ServerState,
) -> Result<Response<Body>, Infallible> {
    if req.uri().path() == "" {
        Ok(Response::builder()
            .header("Location", "/")
            .status(http::StatusCode::TEMPORARY_REDIRECT)
            .body(Body::default())
            .unwrap())
    } else if req.uri().path() == "/robots.txt" {
        Ok(Response::new(Body::from(
            "User-agent: *\r\nDisallow: /\r\n",
        )))
    } else if req.uri().path() == "/map" {
        match handle_map_request(req, state).await {
            Ok(x) => Ok(x),
            Err(e) => Ok(json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &ResponseError {
                    error: format!("{}", e),
                },
            )),
        }
    } else if let Some(body) = HashMap::from(ASSETS).get(req.uri().path()) {
        Ok(Response::new(Body::from(*body)))
    } else if req.uri().path() == "/vecs.json" {
        let data = serde_json::to_string(
            &state
                .emb
                .vecs
                .iter()
                .map(|(name, arr)| (name, dense_matrix_to_json(arr)))
                .collect::<HashMap<_, _>>(),
        )
        .unwrap();
        Ok(Response::builder()
            .header("content-type", "application/json")
            .header("content-disposition", "attachment; filename=\"vecs.json\"")
            .body(Body::from(data))
            .unwrap())
    } else if req.uri().path() == "/api" {
        match handle_api_request(req, state).await {
            Ok(x) => Ok(x),
            Err(e) => Ok(json_response(
                StatusCode::BAD_REQUEST,
                &ResponseError {
                    error: format!("{}", e),
                },
            )),
        }
    } else {
        Ok(Response::builder()
            .status(http::StatusCode::NOT_FOUND)
            .body(Body::from(NOT_FOUND_PAGE))
            .unwrap())
    }
}

async fn handle_api_request(
    req: &Request<Body>,
    state: ServerState,
) -> anyhow::Result<Response<Body>> {
    let x = url::form_urlencoded::parse(req.uri().query().unwrap_or("").as_bytes())
        .collect::<HashMap<_, _>>();
    match x.get("f").map(Deref::deref) {
        Some("knn") => {
            let query = x
                .get("q")
                .ok_or_else(|| anyhow::Error::msg("missing 'q' parameter"))?;
            let count: u32 = x
                .get("count")
                .ok_or_else(|| anyhow::Error::msg("missing 'count' parameter"))?
                .parse()?;
            let results = state.emb.knn(query, count)?;
            Ok(json_response(StatusCode::OK, &results))
        }
        Some("stores") => Ok(json_response(
            StatusCode::OK,
            &StoresResults {
                names: state.emb.store_names.clone(),
                counts: state.emb.store_counts.clone(),
            },
        )),
        Some(_) => Err(anyhow::Error::msg("unexpected value for 'f' parameter: {}")),
        None => Err(anyhow::Error::msg(
            "missing 'f' parameter indicating which function to call",
        )),
    }
}

async fn handle_map_request(
    req: &Request<Body>,
    state: ServerState,
) -> anyhow::Result<Response<Body>> {
    let x = url::form_urlencoded::parse(req.uri().query().unwrap_or("").as_bytes())
        .collect::<HashMap<_, _>>();
    let query = x
        .get("q")
        .ok_or_else(|| anyhow::Error::msg("missing 'q' parameter"))?;
    if let Some(locations) = state.loc_index.lookup(query).await? {
        spawn_blocking(move || encode_map_response(locations, accept_gzip)).await?
    } else {
        Ok(Response::builder()
            .status(http::StatusCode::NOT_FOUND)
            .body(Body::from(NOT_FOUND_PAGE))
            .unwrap())
    }
}

fn encode_map_response(locations: Vec<GeoCoord>) -> anyhow::Result<Response<Body>> {
    let bg_img = base64::encode(WORLD_MAP_JPEG);

    // Don't allow more than one location per pixel, to
    // avoid really dense maps for popular locations.
    let mut dedup_locations = HashMap::new();
    for location in locations {
        let x = 1024.0 * ((location.1 + 180.0) / 360.0);
        let y = 512.0 * ((90.0 - location.0) / 180.0);
        dedup_locations.insert((x.round() as i32, y.round() as i32), (x, y));
    }

    let locations_code: String = dedup_locations
        .values()
        .into_iter()
        .map(|(x, y)| format!("\n    <use xlink:href=\"#dot\" x=\"{}\" y=\"{}\" />", x, y))
        .collect();
    let data_svg = format!(
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>
<svg version=\"1.0\"
 xmlns=\"http://www.w3.org/2000/svg\"
 xmlns:xlink=\"http://www.w3.org/1999/xlink\"
 viewBox=\"0 0 1024 512\">
<defs>
    <g id=\"dot\">
        <circle r=\"5\" fill=\"rgba(255, 0, 0, 0.3)\" />
        <circle r=\"2\" fill=\"rgba(255, 0, 0, 1.0)\" />
    </g>
</defs>
<image width=\"100%\" height=\"100%\"
       xlink:href=\"data:image/jpeg;base64,{}\" />{}
</svg>",
        bg_img, locations_code,
    );
    Ok(Response::builder()
        .header("content-type", "image/svg+xml")
        .body(Body::from(data_svg))
        .unwrap())
}

async fn maybe_compress_response(
    req: &Request<Body>,
    mut resp: Response<Body>,
) -> Result<Response<Body>, Infallible> {
    let accept_gzip = req
        .headers()
        .get("accept-encoding")
        .map(|x| x.to_str().unwrap_or_default())
        .unwrap_or_default()
        .split(", ")
        .map(|x| x.split(";").next().unwrap())
        .any(|x| x == "gzip");
    if !accept_gzip {
        Ok(resp)
    } else {
        let pieces = resp
            .body_mut()
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<hyper::Result<Vec<hyper::body::Bytes>>>();
        Ok(match pieces {
            Err(e) => internal_err_response("compressing body", e),
            Ok(data) => spawn_blocking(move || -> Response<Body> {
                let mut e = GzEncoder::new(Vec::new(), Compression::default());
                for part in data {
                    e.write_all(&part).unwrap();
                }
                let compressed_bytes = e.finish().unwrap();
                let mut builder = Response::builder().header("content-encoding", "gzip");
                for (k, v) in resp.headers().iter() {
                    builder = builder.header(k, v);
                }
                builder.body(Body::from(compressed_bytes)).unwrap()
            })
            .await
            .unwrap_or_else(|e| internal_err_response("compressing body", e)),
        })
    }
}

fn internal_err_response<E: Display>(ctx: &str, err: E) -> Response<Body> {
    eprintln!("internal error: {}: {}", ctx, err);
    Response::builder()
        .status(http::StatusCode::INTERNAL_SERVER_ERROR)
        .body(Body::from(format!("{}: {}", ctx, err)))
        .unwrap()
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

#[derive(Serialize)]
struct KNNResults {
    query: String,
    store_count: u64,
    results: HashMap<String, Vec<String>>,
    dots: HashMap<String, Vec<f32>>,
}

#[derive(Serialize)]
struct StoresResults {
    names: Vec<String>,
    counts: Vec<u64>,
}

#[derive(Clone)]
struct Embeddings {
    vecs: HashMap<String, Array2<f32>>,
    vecs_normalized: HashMap<String, Array2<f32>>,
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
        let mut vecs_normalized = HashMap::new();
        for (name, obj) in all_raw.iter() {
            let matrix = vec_to_matrix(&obj.vecs);
            vecs_normalized.insert(name.clone(), normalize_rows(&matrix));
            vecs.insert(name.clone(), matrix);
        }
        let mut name_to_index = HashMap::new();
        for (i, name) in first.store_names.iter().enumerate() {
            name_to_index.insert(name.clone(), i);
        }
        Ok(Self {
            vecs: vecs,
            vecs_normalized: vecs_normalized,
            name_to_index: name_to_index,
            store_names: first.store_names.clone(),
            store_counts: first.store_counts.clone(),
        })
    }

    fn knn(&self, query: &str, count: u32) -> anyhow::Result<KNNResults> {
        if let Some(idx) = self.name_to_index.get(query) {
            let mut dots = HashMap::new();
            let mut neighbors = HashMap::new();
            for (emb_name, norm_vecs) in self.vecs_normalized.iter() {
                let row_vec = norm_vecs.row(*idx);
                let all_dots = norm_vecs.dot(&row_vec).to_vec();
                let mut sorted_dots: Vec<_> = all_dots.into_iter().enumerate().collect();
                sorted_dots
                    .sort_by(|(_, d1), (_, d2)| d2.partial_cmp(d1).unwrap_or(Ordering::Equal));
                let mut sub_dots = Vec::new();
                let mut sub_names = Vec::new();
                for (idx, dot) in &sorted_dots[0..sorted_dots.len().min(count as usize)] {
                    sub_names.push(self.store_names[*idx].clone());
                    sub_dots.push(*dot);
                }
                dots.insert(emb_name.clone(), sub_dots);
                neighbors.insert(emb_name.clone(), sub_names);
            }
            Ok(KNNResults {
                query: query.to_owned(),
                store_count: self.store_counts[*idx],
                results: neighbors,
                dots: dots,
            })
        } else {
            Err(anyhow::Error::msg("no location found matching the query"))
        }
    }
}

#[derive(Serialize)]
struct ResponseError {
    error: String,
}

fn json_response<T: Serialize>(code: StatusCode, value: &T) -> Response<Body> {
    Response::builder()
        .status(code)
        .header("content-type", "application/json")
        .body(Body::from(match serde_json::to_string_pretty(value) {
            Ok(x) => x,
            Err(e) => serde_json::to_string(&ResponseError {
                error: format!("could not serialize response: {}", e),
            })
            .unwrap(),
        }))
        .unwrap()
}
