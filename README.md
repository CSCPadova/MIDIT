# midit

A Masked Diffusion Transformer for text-to-MIDI generation

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Installation

To install the required dependencies, run the following command:

```bash
make install
```

Or you can install the dependencies manually using uv:

```bash
uv sync
```

## Training

To start training, run the following command:

```bash
uv run train
```

Or run `make train` if you are running the project on a SLURM cluster.

## Aknowledgements

This project is based on the following template repository:

```bibtex
@software{M_Spanio_DEI_Blade_Template,
    author = {M Spanio, Matteo},
    license = {MIT},
    title = {{DEI Blade Template}},
    url = {https://github.com/CSCPadova/dei-blade-template}
}
```

The following libraries were used in this project:

- [PyTorch](https://pytorch.org/) - Deep Learning framework
- [Torchaudio](https://pytorch.org/audio/) - Audio processing library for PyTorch
- [Lightning](https://www.pytorchlightning.ai/) - High-level interface for PyTorch
- [WandB](https://wandb.ai/) - Experiment tracking and visualization
- [Hydra](https://hydra.cc/) - Configuration management

## License

This project template is licensed under the MIT License - see the [LICENSE](LICENSE) file for details, please note that the generated project will inherit this license.
