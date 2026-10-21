- [NviWatch](#nviwatch)
  - [Features](#features)
  - [Installing and Using the Tool](#installing-and-using-the-tool)
    - [Option 1: Download Pre-built Binary](#option-1-download-pre-built-binary)
    - [Option 2: Install via Cargo](#option-2-install-via-cargo)
    - [Option 3: Install via Conda, Mamba, or Pixi](#option-3-install-via-conda-mamba-or-pixi)
    - [Option 4: Build from Source](#option-4-build-from-source)
  - [Usage](#usage)
  - [Key Bindings](#key-bindings)
  - [License](#license)
  - [Contributing](#contributing)
  - [Acknowledgments](#acknowledgments)


# NviWatch

NviWatch is a graphical (gpui-based) application for monitoring NVIDIA GPU devices and processes. Built with Rust, it provides real-time insights into GPU performance metrics, including temperature, utilization, memory usage, and power consumption, with per-process memory attribution and process management.

## Features

- **Real-Time Monitoring**: View real-time data on GPU temperature, utilization, memory usage, and power consumption.
- **Process Management**: Monitor processes running on the GPU and terminate them directly from the interface.
- **Memory Graph**: Stacked per-GPU memory graph separating LLM (compute) processes from other graphics processes.
- **Column Sorting**: Click a column header to sort the process list.
- **Customizable Refresh Rate**: Set the refresh interval for updating GPU metrics.

## Installing and Using the Tool

### Option 1: Download Pre-built Binary

1. Go to the project's GitHub repository.
2. Navigate to the "Releases" section.
3. Download the latest binary release for linux.
4. Once downloaded, open a terminal and navigate to the directory containing the downloaded binary.
5. Make the binary executable with the following command:
   ```
   chmod +x nviwatch
   ```

6. You can now run the tool using:

   ```
   ./nviwatch
   ```

### Option 2: Install via Cargo

If you have Rust and Cargo installed on your system, you can easily install NviWatch directly from crates.io:

1. Open a terminal and run the following command:
   ```bash
   cargo install nviwatch
   ```

2. Once the installation is complete, you can run NviWatch from anywhere:
   ```bash
   nviwatch
   ```

Note: Ensure you have the NVIDIA Management Library (NVML) available on your system before running NviWatch. On Windows, `cargo install nviwatch` builds and runs as well.

### Option 3: Install via Conda, Mamba, or Pixi

NviWatch is also available from [conda-forge](https://anaconda.org/conda-forge/nviwatch):

1. With Conda (after activating an environment):
   ```bash
   conda install -c conda-forge nviwatch
   ```

2. With Mamba (after activating an environment):
   ```bash
   mamba install -c conda-forge nviwatch
   ```

3. With Pixi (user-global install):
   ```bash
   pixi global install --channel conda-forge nviwatch
   ```

4. Once the installation is complete, you can run NviWatch from anywhere:
   ```bash
   nviwatch
   ```

### Option 4: Build from Source

To build and run NviWatch, ensure you have Rust and Cargo installed on your system. You will also need the NVIDIA Management Library (NVML) available.

1. Clone the repository:
   ```bash
   git clone https://github.com/msminhas93/nviwatch.git
   cd nviwatch
   ```

2. Build the project:
   ```bash
   cargo build --release
   ```

3. Run the application:
   ```bash
   chmod +x ./target/release/nviwatch
   ./target/release/nviwatch
   ```

## Usage

NviWatch provides a command-line interface with one option:

- `-w, --watch <MILLISECONDS>`: Set the refresh interval in milliseconds. Default is 300 ms.

**Basic example**:
```bash
# Run with default settings
./nviwatch

# Run with a custom refresh rate
./nviwatch --watch 500
```

## Key Bindings

- **q**: Quit the application
- **↑/↓**: Navigate through the list of processes (mouse click also selects)
- **x**: Terminate the selected process

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgments

- Built with [Rust](https://www.rust-lang.org/) and [gpui](https://gpui.rs).
- Utilizes the [NVIDIA Management Library (NVML)](https://developer.nvidia.com/nvidia-management-library-nvml) via the [nvml_wrapper crate](https://docs.rs/nvml-wrapper/latest/nvml_wrapper/).
