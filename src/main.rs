mod error;
mod gui;
mod utils;

use clap::{Arg, Command};
use nvml::Nvml;

pub(crate) type Result<T> = core::result::Result<T, error::NviError>;

fn main() -> Result<()> {
    let matches = Command::new("nviwatch")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Manpreet Singh")
        .about("NviWatch: A blazingly fast rust based GUI for managing and monitoring NVIDIA GPU processes")
        .arg(
            Arg::new("watch")
                .short('w')
                .long("watch")
                .value_name("MILLISECONDS")
                .help("Refresh interval in milliseconds")
                .default_value("300")
                .required(false),
        )
        .get_matches();

    let watch_interval = matches
        .get_one::<String>("watch")
        .map(|s| s.parse().expect("Invalid number"))
        .unwrap_or(300);

    let nvml = Nvml::init()?;

    gui::run(nvml, watch_interval)
}
