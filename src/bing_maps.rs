use reqwest::Version;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::mem::swap;
use std::pin::Pin;
use std::{
    fmt::{Debug, Display, Write},
    time::Duration,
};
use tokio::time::sleep;

use crate::geo_coord::{GeoBounds, GeoCoord};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    HTTP(reqwest::Error),
    RateLimited,
    RetryLimitExceeded,
    ParseJSON(serde_json::Error),
    ProcessJSON(String),
    ProcessHTML(String),
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HTTP(e) => write!(f, "error making request: {}", e),
            Self::RateLimited => write!(f, "rate limit exceeded"),
            Self::RetryLimitExceeded => write!(f, "request retry limit exceeded"),
            Self::ParseJSON(e) => write!(f, "error parsing JSON: {}", e),
            Self::ProcessJSON(e) => write!(f, "error processing JSON structure: {}", e),
            Self::ProcessHTML(e) => write!(f, "error processing html structure: {}", e),
        }
    }
}

impl std::error::Error for Error {}

impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        Error::HTTP(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::ParseJSON(e)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MapItem {
    pub id: String,
    pub name: String,
    pub location: GeoCoord,
    pub address: Option<String>,
    pub phone: Option<String>,
    pub chain_id: Option<String>,
    pub category_path: Option<String>,
    pub category_name: Option<String>,
}

impl MapItem {
    fn from_json(parsed: &Value) -> Result<MapItem> {
        Ok(MapItem {
            id: read_object(parsed, "entity.id")?,
            name: read_object(parsed, "entity.title")?,
            location: GeoCoord(
                read_object(parsed, "geometry.x")?,
                read_object(parsed, "geometry.y")?,
            ),
            address: read_object(parsed, "entity.address").ok(),
            phone: read_object(parsed, "entity.phone").ok(),
            chain_id: read_object(parsed, "entity.chainId").ok(),
            category_path: read_object(parsed, "entity.primaryCategoryPath").ok(),
            category_name: read_object(parsed, "entity.primaryCategoryName").ok(),
        })
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Tile {
    pub level_of_detail: u8,
    pub x: u32,
    pub y: u32,
}

impl Tile {
    pub fn new(lod: u8, x: u32, y: u32) -> Tile {
        assert!(lod > 0 && lod < 24);
        assert!(x < (1 << lod));
        assert!(y < (1 << lod));
        Tile {
            level_of_detail: lod,
            x: x,
            y: y,
        }
    }

    pub fn all_tiles(lod: u8) -> Vec<Tile> {
        let mut res = Vec::new();
        for x in 0..(1 << lod) {
            for y in 0..(1 << lod) {
                res.push(Tile::new(lod, x, y));
            }
        }
        res
    }

    pub fn parent(&self) -> Option<Tile> {
        if self.level_of_detail == 1 {
            None
        } else {
            Some(Tile {
                level_of_detail: self.level_of_detail - 1,
                x: self.x / 2,
                y: self.y / 2,
            })
        }
    }

    pub fn children(&self) -> [Tile; 4] {
        let lod = self.level_of_detail + 1;
        [
            Tile {
                level_of_detail: lod,
                x: self.x << 1,
                y: self.y << 1,
            },
            Tile {
                level_of_detail: lod,
                x: (self.x << 1) + 1,
                y: self.y << 1,
            },
            Tile {
                level_of_detail: lod,
                x: self.x << 1,
                y: (self.y << 1) + 1,
            },
            Tile {
                level_of_detail: lod,
                x: (self.x << 1) + 1,
                y: (self.y << 1) + 1,
            },
        ]
    }

    pub fn children_at_lod(&self, level_of_detail: u8, out: &mut Vec<Tile>) {
        assert!(self.level_of_detail <= level_of_detail);
        if self.level_of_detail == level_of_detail {
            out.push(self.clone());
        } else {
            for child in self.children() {
                child.children_at_lod(level_of_detail, out);
            }
        }
    }

    pub fn quadkey(&self) -> String {
        let end = char::from_digit(((self.y & 1) << 1) | (self.x & 1), 4).unwrap();
        match self.parent() {
            Some(parent) => {
                let mut res = parent.quadkey();
                res.write_char(end).unwrap();
                res
            }
            None => end.into(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PointOfInterest {
    pub id: String,
    pub name: String,
    pub location: GeoCoord,
}
pub struct Client {
    client: reqwest::Client,
}

impl Client {
    pub fn new() -> Client {
        Client {
            client: reqwest::Client::new(),
        }
    }

    pub async fn map_search(
        &mut self,
        query: &str,
        bounds: &GeoBounds,
        max_retries: u32,
    ) -> Result<Vec<MapItem>> {
        self.retry_loop(max_retries, (query, bounds), |cli, (query, bounds)| {
            Box::pin(cli.map_search_attempt(*query, *bounds))
        })
        .await
    }

    async fn map_search_attempt(&self, query: &str, bounds: &GeoBounds) -> Result<Vec<MapItem>> {
        let response = self
            .client
            .get("https://www.bing.com/maps/overlaybfpr")
            .version(Version::HTTP_11)
            .query(&[
                ("q", query),
                ("filters", "direction_partner:\"maps\""),
                ("mapcardtitle", ""),
                ("p1", "[AplusAnswer]"),
                ("count", "100"),
                ("ecount", "100"),
                ("first", "0"),
                ("efirst", "1"),
                (
                    "localMapView",
                    &format!(
                        "{:.15},{:.15},{:.15},{:.15}",
                        bounds.1 .0, bounds.0 .1, bounds.0 .0, bounds.1 .1
                    ),
                ),
                ("ads", "0"),
                (
                    "cp",
                    &format!("{:.15}~{:.15}", bounds.mid().0, bounds.mid().1),
                ),
            ])
            .timeout(Duration::from_secs(30))
            .send()
            .await?
            .text()
            .await?;

        // When overloaded, the server responds with messages of the form:
        // Ref A: DC5..................73B Ref B: AMB......06 Ref C: 2022-09-20T00:20:31Z
        if response.starts_with("Ref A:") {
            return Err(Error::RateLimited);
        }

        let doc = Html::parse_fragment(&response);
        let mut result = Vec::new();
        for obj in doc.select(&Selector::parse("a.listings-item").unwrap()) {
            if let Some(info_json) = obj.value().attr("data-entity") {
                let parsed: Value = serde_json::from_str(info_json)?;
                result.push(MapItem::from_json(&parsed)?);
            }
        }
        Ok(result)
    }

    pub async fn points_of_interest(
        &mut self,
        tile: &Tile,
        category_id: &str,
        query: Option<&str>,
        chain_id: Option<&str>,
        max_retries: u32,
    ) -> Result<Vec<PointOfInterest>> {
        self.retry_loop(
            max_retries,
            (tile, category_id, query, chain_id),
            |cli, (tile, category_id, query, chain_id)| {
                Box::pin(cli.points_of_interest_attempt(*tile, *category_id, *query, *chain_id))
            },
        )
        .await
    }

    async fn points_of_interest_attempt(
        &self,
        tile: &Tile,
        category_id: &str,
        query: Option<&str>,
        chain_id: Option<&str>,
    ) -> Result<Vec<PointOfInterest>> {
        let quadkey = tile.quadkey();
        let raw_response = self
            .client
            .get("https://www.bingapis.com/api/v7/micropoi")
            .version(Version::HTTP_11)
            .query(&[
                ("tileid", quadkey.as_ref()),
                ("q", query.unwrap_or("")),
                ("chainid", chain_id.unwrap_or("")),
                ("categoryid", category_id),
                ("appid", "5BA026015AD3D08EF01FBD643CF7E9061C63A23B"),
            ])
            .timeout(Duration::from_secs(30))
            .send()
            .await?;
        if !raw_response.status().is_success() {
            return Err(Error::RateLimited);
        }
        let response = raw_response.text().await?;
        let parsed: Value = serde_json::from_str(&response)?;
        if parsed.is_object() && !parsed.as_object().unwrap().contains_key("results") {
            Ok(Vec::new())
        } else {
            let results: Vec<Value> = read_object(&parsed, "results")?;
            results
                .into_iter()
                .map(|x| -> Result<PointOfInterest> {
                    Ok(PointOfInterest {
                        id: read_object(&x, "id")?,
                        name: read_object(&x, "name")?,
                        location: GeoCoord(
                            read_object(&x, "geo.latitude")?,
                            read_object(&x, "geo.longitude")?,
                        ),
                    })
                })
                .collect()
        }
    }

    pub async fn id_lookup(&mut self, ypid: &str, max_retries: u32) -> Result<Option<MapItem>> {
        self.retry_loop(max_retries, ypid, |cli, ypid| {
            Box::pin(cli.id_lookup_attempt(ypid))
        })
        .await
    }

    async fn id_lookup_attempt(&self, ypid: &str) -> Result<Option<MapItem>> {
        let response = self
            .client
            .get("https://www.bing.com/maps/infoboxoverlaybfpr")
            .version(Version::HTTP_11)
            .query(&[
                ("q", ""),
                (
                    "filters",
                    &format!("local_ypid:\"{}\" direction_partner:\"maps\"", ypid),
                ),
            ])
            .timeout(Duration::from_secs(30))
            .send()
            .await?
            .text()
            .await?;

        // When overloaded, the server responds with messages of the form:
        // Ref A: DC5..................73B Ref B: AMB......06 Ref C: 2022-09-20T00:20:31Z
        if response.starts_with("Ref A:") {
            return Err(Error::RateLimited);
        }

        let doc = Html::parse_fragment(&response);

        // When the query has no results, an error is present in the response.
        if doc
            .select(&Selector::parse("div.mobileErrMsgFail").unwrap())
            .count()
            > 0
        {
            return Ok(None);
        }

        for obj in doc.select(&Selector::parse("div.overlay-taskpane").unwrap()) {
            if let Some(info_json) = obj.value().attr("data-entity") {
                let parsed: Value = serde_json::from_str(info_json)?;
                return Ok(Some(MapItem::from_json(&parsed)?));
            }
        }
        Err(Error::ProcessHTML(
            "could not find data-entity in results".to_owned(),
        ))
    }

    async fn retry_loop<'a, T: Debug, F, A>(&'a mut self, max_retries: u32, a: A, f: F) -> Result<T>
    where
        A: 'a,
        // https://stackoverflow.com/questions/70746671/how-to-bind-lifetimes-of-futures-to-fn-arguments-in-rust
        for<'b> F: Fn(&'b Self, &'b A) -> Pin<Box<dyn Future<Output = Result<T>> + Send + 'b>>,
    {
        let mut remaining_tries = max_retries + 1;
        loop {
            let mut last_result = Err(Error::RetryLimitExceeded);
            for retry_timeout in [0.1, 1.0, 2.0, 4.0, 8.0, 10.0, 16.0, 32.0] {
                match f(self, &a).await {
                    res @ Ok(_) => return res,
                    Err(Error::RateLimited) => sleep(Duration::from_secs_f64(retry_timeout)).await,
                    res @ Err(_) => {
                        last_result = res;
                        break;
                    }
                }
            }
            last_result = self.reset_client_on_err(last_result);
            remaining_tries -= 1;
            if remaining_tries == 0 {
                return last_result;
            }
            eprintln!("retrying after error: {}", last_result.unwrap_err());
            sleep(Duration::from_secs(10)).await;
        }
    }

    fn reset_client_on_err<T>(&mut self, res: Result<T>) -> Result<T> {
        match res {
            x @ Err(Error::HTTP(_)) => {
                swap(&mut self.client, &mut reqwest::Client::new());
                x
            }
            x => x,
        }
    }
}

fn read_object<T: FromJSON>(root: &Value, path: &str) -> Result<T> {
    let mut cur_obj = root;
    for part in path.split(".") {
        if let Value::Object(obj) = cur_obj {
            if let Some(x) = obj.get(part) {
                cur_obj = x;
            } else {
                return Err(Error::ProcessJSON(format!(
                    "object path not found: {}",
                    path
                )));
            }
        } else {
            return Err(Error::ProcessJSON(format!(
                "incorrect type in object path: {}",
                path
            )));
        }
    }
    match T::from_json(cur_obj) {
        Ok(x) => Ok(x),
        Err(Error::ProcessJSON(x)) => Err(Error::ProcessJSON(format!(
            "error for object path {}: {}",
            path, x
        ))),
        other => other,
    }
}

trait FromJSON
where
    Self: Sized,
{
    fn from_json(value: &Value) -> Result<Self>;
}

impl FromJSON for f64 {
    fn from_json(value: &Value) -> Result<Self> {
        match value {
            Value::Number(x) => {
                if let Some(f) = x.as_f64() {
                    Ok(f)
                } else {
                    Err(Error::ProcessJSON(format!("{} is not an f64", x)))
                }
            }
            _ => Err(Error::ProcessJSON(format!("{} is not a number", value))),
        }
    }
}

impl FromJSON for String {
    fn from_json(value: &Value) -> Result<Self> {
        match value {
            Value::String(x) => Ok(x.clone()),
            _ => Err(Error::ProcessJSON(format!("{} is not a string", value))),
        }
    }
}

impl FromJSON for Vec<Value> {
    fn from_json(value: &Value) -> Result<Self> {
        match value {
            Value::Array(x) => Ok(x.clone()),
            _ => Err(Error::ProcessJSON(format!("{} is not an array", value))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Tile;

    #[test]
    fn test_tile_quadkey() {
        let items = [(Tile::new(3, 3, 5), "213"), (Tile::new(2, 1, 2), "21")];
        for (tile, expected) in items {
            assert_eq!(tile.quadkey(), expected);
        }
    }
}
