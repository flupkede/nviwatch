#![cfg_attr(windows, windows_subsystem = "windows")]

mod error;
mod gui;
mod utils;

use clap::{Arg, Command};
use nvml::Nvml;

pub(crate) type Result<T> = core::result::Result<T, error::NviError>;

fn main() {
    if let Err(e) = run() {
        report(&format!("nviwatch: {e}"));
        std::process::exit(1);
    }
}

/// With the windows GUI subsystem there is no console, so every failure
/// path reports through a message box instead of stderr.
#[cfg(windows)]
fn report(text: &str) {
    use windows::Win32::UI::WindowsAndMessaging::{MB_ICONERROR, MessageBoxW};
    use windows::core::HSTRING;
    let text = HSTRING::from(text);
    let caption = HSTRING::from("nviwatch");
    unsafe {
        let _ = MessageBoxW(None, &text, &caption, MB_ICONERROR);
    }
}

#[cfg(not(windows))]
fn report(text: &str) {
    eprintln!("{text}");
}

fn run() -> Result<()> {
    let matches = matches();

    let watch_interval = matches.get_one::<u64>("watch").copied().unwrap_or(300);

    let nvml = Nvml::init()?;

    gui::run(nvml, watch_interval)
}

fn matches() -> clap::ArgMatches {
    let cmd = Command::new("nviwatch")
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
                .value_parser(clap::value_parser!(u64))
                .required(false),
        );
    match cmd.try_get_matches() {
        Ok(m) => m,
        Err(e) => {
            report(&e.to_string());
            std::process::exit(e.exit_code());
        }
    }
}
