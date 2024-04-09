# HMCMT: Hamiltonian Monte Carlo for 2D Magnetotelluric (MT) Data Inversion

The HMCMT package is a sophisticated tool designed for the 2D probabilistic inversion of Magnetotelluric (MT) data, coupled with uncertainty quantification using the Hamiltonian Monte Carlo method. Developed with the high-performance [Julia programming language](http://julialang.org), HMCMT facilitates advanced geophysical research by enabling precise analysis and interpretation of MT data.

## License

HMCMT is freely available under the GNU General Public License. For more details, please refer to the [license documentation](http://www.gnu.org/licenses/).

## File Structure

- **./doc:** Documentation providing file format instructions.
- **./examples:** Contains synthetic and field examples showcased in the manuscript, organized into subdirectories.
- **./src:** The source code of HMCMT.

## Installation Instructions

### For Julia

HMCMT is compatible with Julia v1.0 and subsequent releases.

#### Windows

1. Visit the [Julia download page](https://julialang.org/downloads/) and download the Windows version (.exe).
2. Follow the installation prompts to complete the installation.

#### Linux

While Julia is cross-platform, running HMCMT on Linux is recommended for enhanced compatibility, especially with third-party packages such as MUMPS. 

There are several methods to install Julia on Linux:

- **Precompiled Binaries (Recommended):** Download the generic Linux binaries (.tar.gz file) from the [Julia download page](https://julialang.org/downloads/). Ensure Julia's executable is in your system's PATH by extracting the .tar.gz file to a preferred location and creating a symbolic link or directly adding Julia’s bin folder to your PATH.

- **Compiling from Source:** Clone the Julia repository from GitHub and compile it. This method requires Git, g++, gfortran, and m4. Follow the compilation instructions provided in the repository's README.

- **PPA for Ubuntu Linux:** Ubuntu users (12.04 and later) can install Julia using the Personal Package Archive (PPA), simplifying the installation process through a series of terminal commands.

  ```shell
  sudo add-apt-repository ppa:staticfloat/juliareleases
  sudo add-apt-repository ppa:staticfloat/julia-deps
  sudo apt update
  sudo apt install julia
  ```

- After a successful installation, Julia can be started by double-clicking the Julia executable (on Windows) or typing `julia` from the command line (on Linux). Following is the Julia's command line environment (the so-called REPL):

  ```shell
     _       _ _(_)_     |
    (_)     | (_) (_)    |  Documentation: https://docs.julialang.org
    _  _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
    | | | | | | |/ _` |  |
    | | |_| | | | (_| |  |  Version 1.10.0 (2023-12-25)
   _/ |\__'_|_|_|\__'_|  |  Official http://julialang.org/ release
  |__/                   |  
  
  julia>
  ```

### Special Note on MUMPS Integration

Due to the specialized use of the MUMPS package within our environment and the official version's limitations, particularly on Windows systems, we provide a custom-compiled MUMPS package. This ensures compatibility and enhances performance for our users.

1. **Custom MUMPS Package:** Initially based on an early version of the MUMPS package for Julia, we have incorporated the latest enhancements into our provided package to ensure seamless integration and superior performance.
2. **Windows Compatibility:** Recognizing the challenges of running MUMPS on Windows, we have compiled a Windows-specific version of MUMPS.dll using MSYS2 and gfortran. This enables Windows users to enjoy the same level of efficiency and reliability as their Linux counterparts.

## Running HMCMT

### Setting Up the Environment

HMCMT requires several external dependencies, which can be easily installed through Julia's package manager (Pkg) by activating and instantiating the package environment. Detailed instructions are provided on how to set up your environment to run HMCMT successfully.

```shell
cd /home/username/code/HMCMT
```

And then, enter the Julia REPL. Then press `]` from the Julia REPL you will enter the Pkg REPL which looks like
```julia
(v1.10) pkg>
```

Now you are currently in the environment named v1.10, Julia 1.10's default environment. To switch to the package environment, just `activate` the current directory:
```julia
(v1.10) pkg> activate .
```
Then you will get:
```julia
(HMCMT) pkg>
```
Until now you are in the environment HMCMT. The environment will not be well-configured until you `instantiate` it:
```julia
(HMCMT) pkg> instantiate
```
By doing so the dependencies listed in `Project.toml` and `Manifest.toml` can be automatically downloaded and installed.

### Running the Code

* First, you need to let the **HMCMT** package to be "loaded" by the current Julia environment. This is done by adding the parent directory of the package directory to  `LOAD_PATH`, a global environment variable of Julia. For example, the HMCMT package is placed at `home/username/code` on Linux or at `D:\\code` on Windows, then type the following command from the Julia REPL:

  **On Linux:**
  
  ```julia
  julia> push!(LOAD_PATH,"/home/username/code")
  ```
  **On Windows:** 
  
  ```julia
  julia> push!(LOAD_PATH,"D:\\code")
  ```
  
  


* Finally, go to the directory where the running script loaded, and run the script by typing the following command (for example) from the Julia REPL:

  ```julia
  julia> include("runHMCscript.jl")
  ```

### Parallel HMC Sampling
To perform parallel MCMC sampling, call the parallel HMC sampling function `parallelHMCSampler` instead of  the single HMC sampler `runHMCSampler` (please refer to the `paraHMCScript.jl` scripts within the `examples` directory).

* first you need to launch multiple worker processes by either starting Julia like

  ```shell
  shell> julia -p 4
  ```

  or adding processes within Julia (recommended) like

  ```julia
  julia> addprocs(4)
  ```

* Then you need to get the **HMCMT** package to be "loaded" on all processes by following command from the Julia REPL:

  ```julia
  julia> @everywhere push!(LOAD_PATH,"/home/username/code")
  ````

* Finally, go to the directory where the running script loaded, and run the script by typing the following command (for example) from the Julia REPL:

  ```julia
  julia> include("paraHMCScript.jl")
  ```

### Writing Your Own Running Script

In the `./examples` directory, we have supplied well-documented scripts for both serial and parallel HMC inversion process, named `runHMCScript.jl` or `paraHMCScript.jl`, respectively. These scripts serve as a foundation upon which users can build. We encourage users to customize these scripts according to their unique research needs, promoting both flexibility and adaptability across a range of geophysical inversion and uncertainty quantification projects.

Adhering to these guidelines and leveraging the custom resources we provide will enable users to fully harness the capabilities of HMCMT for their specific geophysical inversion and uncertainty quantification endeavors.
