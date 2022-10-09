mod bing_maps;
mod clean;
mod cooccurrence;
mod discover;
mod geo_coord;
mod scrape;
mod website;

use clap::Parser;
use std::process::ExitCode;

#[derive(Parser, Clone)]
#[clap(author, version, about, long_about = None)]
enum Cli {
    Scrape {
        #[clap(flatten)]
        args: scrape::ScrapeArgs,
    },
    Discover {
        #[clap(flatten)]
        args: discover::DiscoverArgs,
    },
    Clean {
        #[clap(flatten)]
        args: clean::CleanArgs,
    },
    Cooccurrence {
        #[clap(flatten)]
        args: cooccurrence::CoocurrenceArgs,
    },
    Website {
        #[clap(flatten)]
        args: website::WebsiteArgs,
    },
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Err(e) = match cli {
        Cli::Clean { args } => clean::clean(args).await,
        Cli::Cooccurrence { args } => cooccurrence::cooccurrence(args).await,
        Cli::Discover { args } => discover::discover(args).await,
        Cli::Scrape { args } => scrape::scrape(args).await,
        Cli::Website { args } => website::website(args).await,
    } {
        eprintln!("{}", e);
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
